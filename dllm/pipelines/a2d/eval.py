"""
accelerate launch \
    --num_processes 4 \
    dllm/pipelines/a2d/eval.py \
    --tasks gsm8k_cot \
    --model bd3lm \
    --apply_chat_template \
    --num_fewshot 5 \
    --model_args "pretrained=dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1,max_new_tokens=256,steps=256,block_size=32,cfg=0.0"
"""

from dataclasses import dataclass
from types import SimpleNamespace

import accelerate
import torch
import torch.nn.functional as F
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from tqdm import tqdm

import dllm
from dllm.core.samplers import BD3LMSampler, BD3LMSamplerConfig
from dllm.pipelines.llada.eval import LLaDAEvalConfig, LLaDAEvalHarness


@dataclass
class MDLMEvalConfig(LLaDAEvalConfig):
    """Alias config for A2D MDLM eval (inherits LLaDA settings)."""


@register_model("mdlm")
class MDLMEvalHarness(LLaDAEvalHarness):
    def __init__(self, config: MDLMEvalConfig | None = None, **kwargs):
        if config is None:
            config = MDLMEvalConfig()
        super().__init__(config=config, **kwargs)


@dataclass
class BD3LMEvalConfig(BD3LMSamplerConfig):
    max_new_tokens: int = 128
    max_length: int = 2048
    steps: int = 128
    block_size: int = 32

    pretrained: str = ""
    dtype: str | torch.dtype = "auto"
    batch_size: int = 32
    mc_num: int = 128
    is_check_greedy: bool = False
    device: str = "cuda"


@register_model("bd3lm")
class BD3LMEvalHarness(LM):
    def __init__(
        self,
        config: BD3LMEvalConfig | None = None,
        **kwargs,
    ):
        super().__init__()

        # Initialize config if not provided
        if config is None:
            config = BD3LMEvalConfig()

        # Pull args from config, allow kwargs to override
        pretrained = kwargs.get("pretrained", config.pretrained)
        dtype = kwargs.get("dtype", config.dtype)
        batch_size = kwargs.get("batch_size", config.batch_size)
        mc_num = kwargs.get("mc_num", config.mc_num)
        is_check_greedy = kwargs.get("is_check_greedy", config.is_check_greedy)
        device = kwargs.get("device", config.device)
        cfg = kwargs.get("cfg", config.cfg_scale)
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        block_size = kwargs.get("block_size", config.block_size)
        max_length = kwargs.get("max_length", config.max_length)
        remasking = kwargs.get("remasking", config.remasking)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)

        accelerator = accelerate.Accelerator()

        # Get GLOBAL rank from torch.distributed (not accelerator)
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()  # ← GLOBAL rank (0-15)
            self._world_size = (
                torch.distributed.get_world_size()
            )  # ← GLOBAL world size (16)
        else:
            self._rank = 0
            self._world_size = 1

        # Use accelerator for device placement
        self.model = dllm.utils.get_model(
            SimpleNamespace(model_name_or_path=pretrained, dtype=get_dtype(dtype))
        )
        self.model.eval()

        if accelerator.num_processes > 1:
            # Let accelerator handle device placement
            self.model = accelerator.prepare(self.model)
            self.device = (
                accelerator.device
            )  # ← Accelerator figures out local device correctly
            self.accelerator = accelerator
        else:
            # Single GPU
            self.model = self.model.to(device)
            self.device = torch.device(device)
            self.accelerator = None

        self.tokenizer = dllm.utils.get_tokenizer(
            SimpleNamespace(model_name_or_path=pretrained, model=self.model)
        )

        # sampler params
        self.mask_id = self.tokenizer.mask_token_id
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.max_new_tokens = int(max_new_tokens)
        self.block_size = int(block_size)
        self.steps = int(steps)
        self.cfg = float(cfg)
        self.remasking = remasking
        self.is_check_greedy = is_check_greedy
        self.right_shift_logits = right_shift_logits

        # loglikelihood params
        self.mc_num = int(mc_num)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.0

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def generate_until(self, requests: list[Instance]):
        """Generate greedily until a stopping sequence

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, gen_kwargs).
            context: str
                Context string
            gen_kwargs: dict
                A dictionary of keyword arguments to pass to the generation function e.g. top_k, until, etc.
        :return: list[str]
            A list of model generated continuations.
            continuation: str
                The generated continuation.
        """
        out = []
        sampler = BD3LMSampler(model=self.model, tokenizer=self.tokenizer)

        for instance in tqdm(requests, desc="Generating..."):
            context, gen_kwargs = instance.args  # type: ignore
            prompt_ids = torch.tensor(
                self.tokenizer(context)["input_ids"],
                device=self.device,
                dtype=torch.long,
            )
            prompt = [prompt_ids]
            stop_tokens = gen_kwargs["until"]
            generated_ids = sampler.sample(
                inputs=prompt,
                steps=self.steps,
                max_new_tokens=self.max_new_tokens,
                block_size=self.block_size,
                temperature=0.0,
                cfg_scale=self.cfg,
                remasking=self.remasking,
                right_shift_logits=self.right_shift_logits,
            )
            generated_answer = self.tokenizer.decode(
                generated_ids[0][prompt[0].shape[0] :], skip_special_tokens=False
            )
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(
                generated_answer_ids, skip_special_tokens=True
            )
            out.append(generated_answer)
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out

    def loglikelihood(self, requests: list[Instance]):
        raise NotImplementedError("loglikelihood not supported for this model")

    def loglikelihood_rolling(self, requests: list[Instance]):
        raise NotImplementedError("loglikelihood_rolling not supported for this model")


if __name__ == "__main__":
    cli_evaluate()
