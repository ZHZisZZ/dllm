"""
accelerate launch \
    --num_processes 4 \
    dllm/pipelines/bert/eval.py \
    --tasks gsm8k_bert \
    --model bert \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=dllm-collection/ModernBERT-base-chat-v0.1,max_new_tokens=256,steps=256,block_size=32"
"""

from dataclasses import dataclass

import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness


@dataclass
class BERTEvalConfig(MDLMEvalConfig):
    max_new_tokens: int = 128
    max_length: int = 512
    steps: int = 128
    block_size: int = 128

    pretrained: str = ""
    dtype: str | torch.dtype = "auto"
    batch_size: int = 32
    mc_num: int = 128
    is_check_greedy: bool = False
    device: str = "cuda"


@register_model("bert")
class BERTEvalHarness(MDLMEvalHarness):
    def __init__(
        self,
        config: BERTEvalConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = BERTEvalConfig()
        super().__init__(config=config, **kwargs)

    def _prepare_prompt_for_generation(self, context: str) -> list[torch.Tensor]:
        """Drop [CLS] and [SEP] from BERT tokenizer output for generation."""
        prompt_ids = torch.tensor(
            self.tokenizer(context)["input_ids"],
            device=self.device,
            dtype=torch.long,
        )
        trimmed_prompt = (
            prompt_ids[1:-1] if prompt_ids.numel() > 2 else prompt_ids
        )
        return [trimmed_prompt]


if __name__ == "__main__":
    cli_evaluate()
