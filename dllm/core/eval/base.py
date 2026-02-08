"""
Generic eval harness base: accelerator, rank/world_size, model/tokenizer loading,
device, apply_chat_template, tokenizer_name. Pipeline-agnostic; no MDLM/Dream specifics.

Run: Not runnable directly; use pipeline eval entrypoints (e.g. dllm.pipelines.llada.eval).
"""

from dataclasses import dataclass
from types import SimpleNamespace

import accelerate
import torch
from lm_eval.api.model import LM
from lm_eval.models.utils import get_dtype

import dllm


@dataclass
class BaseEvalConfig:
    """Minimal config for base eval: model loading and device only."""

    pretrained: str = ""
    dtype: str | torch.dtype = "auto"
    device: str = "cuda"
    resolve_pretrained_with_base_env: bool = True
    """If True, resolve pretrained path with dllm.utils.resolve_with_base_env(BASE_MODELS_DIR)."""


class BaseEvalHarness(LM):
    """
    Pipeline-agnostic eval base: accelerator, rank/world_size, model and tokenizer
    loading, device placement, apply_chat_template, tokenizer_name.
    Subclasses implement loglikelihood, loglikelihood_rolling, generate_until.
    """

    def __init__(
        self,
        config: BaseEvalConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        if config is None:
            config = BaseEvalConfig()

        pretrained = kwargs.get("pretrained", config.pretrained)
        dtype = kwargs.get("dtype", config.dtype)
        device = kwargs.get("device", config.device)
        resolve_base = kwargs.get(
            "resolve_pretrained_with_base_env", config.resolve_pretrained_with_base_env
        )

        if resolve_base and pretrained:
            pretrained = dllm.utils.resolve_with_base_env(pretrained, "BASE_MODELS_DIR")

        accelerator = accelerate.Accelerator()

        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        # Optional: subclasses can provide get_model_config(pretrained) for custom model config (backward compatible).
        model_config = None
        if hasattr(config, "get_model_config") and callable(getattr(config, "get_model_config")):
            model_config = config.get_model_config(pretrained)

        self.model = dllm.utils.get_model(
            SimpleNamespace(model_name_or_path=pretrained, dtype=get_dtype(dtype)),
            config=model_config,
        )
        self.model.eval()

        if accelerator.num_processes > 1:
            self.model = accelerator.prepare(self.model)
            self.device = accelerator.device
            self.accelerator = accelerator
        else:
            self.model = self.model.to(device)
            self.device = torch.device(device)
            self.accelerator = None

        self.tokenizer = dllm.utils.get_tokenizer(
            SimpleNamespace(model_name_or_path=pretrained, model=self.model)
        )

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format chat history for input to the LM."""
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    def _encode_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        """
        Encode context and continuation; move trailing spaces from context to continuation.
        Subclasses may override (e.g. add_bos_token, eos, truncation).
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        raise NotImplementedError
