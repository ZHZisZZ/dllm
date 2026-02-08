"""
accelerate launch \
    --num_processes 4 \
    dllm/pipelines/fastdllm/llada/eval.py \
    --tasks gsm8k_cot \
    --model fastdllm_llada \
    --apply_chat_template \
    --num_fewshot 5 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,max_new_tokens=512,steps=512,block_size=512"
"""

from dataclasses import dataclass

import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalHarness
from dllm.pipelines.fastdllm.llada import (
    FastdLLMLLaDAConfig,
    FastdLLMLLaDASampler,
    FastdLLMLLaDASamplerConfig,
)


def _parse_token_list(value):
    """Parse token list from string format like '[126081;126348]' or list."""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        if not value:
            return []
        return [int(x.strip()) for x in value.split(";") if x.strip()]
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return []


@dataclass
class FastdLLMLLaDAEvalConfig(FastdLLMLLaDASamplerConfig):
    # According to LLaDA's opencompass implementation: https://github.com/ML-GSAI/LLaDA/blob/main/opencompass/opencompass/models/dllm.py
    max_new_tokens: int = 1024
    max_length: int = 4096
    steps: int = 1024
    block_size: int = 1024

    pretrained: str = ""
    dtype: str | torch.dtype = "auto"
    batch_size: int = 32
    mc_num: int = 128
    is_check_greedy: bool = False
    device: str = "cuda"
    use_cache: str | None = None  # prefix | dual | None
    threshold: float | None = None
    factor: float | None = None
    cfg_scale: float = 0.0  # for MDLMEvalHarness compat; unused by FastdLLM LLaDA
    resolve_pretrained_with_base_env: bool = True

    def get_model_config(self, pretrained: str):
        """Return FastdLLM model config so BaseEvalHarness loads the correct model."""
        return FastdLLMLLaDAConfig.from_pretrained(pretrained)


@register_model("fastdllm_llada")
class FastdLLMLLaDAEvalHarness(MDLMEvalHarness):
    """LLaDA eval harness for FastdLLM LLaDA model; inherits from MDLMEvalHarness."""

    def __init__(
        self,
        config: FastdLLMLLaDAEvalConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = FastdLLMLLaDAEvalConfig()
        super().__init__(config=config, **kwargs)
        use_cache = kwargs.get("use_cache", config.use_cache)
        threshold = kwargs.get("threshold", config.threshold)
        factor = kwargs.get("factor", config.factor)
        suppress_tokens = _parse_token_list(
            kwargs.get("suppress_tokens", config.suppress_tokens)
        )
        begin_suppress_tokens = _parse_token_list(
            kwargs.get("begin_suppress_tokens", config.begin_suppress_tokens)
        )
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        self.use_cache = use_cache
        self.threshold = threshold
        self.factor = factor
        self.suppress_tokens = suppress_tokens
        self.begin_suppress_tokens = begin_suppress_tokens
        self.right_shift_logits = right_shift_logits

    def _create_sampler(self):
        return FastdLLMLLaDASampler(model=self.model, tokenizer=self.tokenizer)

    def _get_sampler_kwargs(self):
        return {
            "use_cache": self.use_cache,
            "threshold": self.threshold,
            "factor": self.factor,
            "suppress_tokens": self.suppress_tokens,
            "begin_suppress_tokens": self.begin_suppress_tokens,
            "right_shift_logits": self.right_shift_logits,
        }


if __name__ == "__main__":
    cli_evaluate()
