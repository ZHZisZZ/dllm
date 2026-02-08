"""
accelerate launch \
    --num_processes 4 \
    dllm/pipelines/llada/eval.py \
    --tasks gsm8k_cot \
    --model llada \
    --apply_chat_template \
    --num_fewshot 5 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,max_new_tokens=512,steps=512,block_size=512,cfg_scale=0.0"
"""

from dataclasses import dataclass

import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness


@dataclass
class LLaDAEvalConfig(MDLMEvalConfig):
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


@register_model("llada")
class LLaDAEvalHarness(MDLMEvalHarness):
    @staticmethod
    def _parse_token_list(value):
        """Parse token list from string format like '[126081;126348]' or list."""
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]  # Remove brackets
            if not value:  # Empty string after removing brackets
                return []
            return [int(x.strip()) for x in value.split(";") if x.strip()]
        elif isinstance(value, list):
            return value
        elif value is None:
            return []
        return []

    def __init__(
        self,
        config: LLaDAEvalConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = LLaDAEvalConfig()

        super().__init__(config=config, **kwargs)

        # LLaDA-specific: suppress_tokens, begin_suppress_tokens, right_shift_logits
        self.suppress_tokens = self._parse_token_list(
            kwargs.get("suppress_tokens", config.suppress_tokens)
        )
        self.begin_suppress_tokens = self._parse_token_list(
            kwargs.get("begin_suppress_tokens", config.begin_suppress_tokens)
        )
        self.right_shift_logits = kwargs.get(
            "right_shift_logits", config.right_shift_logits
        )

    def _get_sampler_kwargs(self) -> dict:
        return {
            "suppress_tokens": self.suppress_tokens,
            "begin_suppress_tokens": self.begin_suppress_tokens,
            "right_shift_logits": self.right_shift_logits,
        }


if __name__ == "__main__":
    cli_evaluate()
