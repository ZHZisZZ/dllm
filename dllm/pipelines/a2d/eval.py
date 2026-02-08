"""
accelerate launch \
    --num_processes 4 \
    dllm/pipelines/a2d/eval.py \
    --tasks gsm8k_cot \
    --model a2d_mdlm \
    --apply_chat_template \
    --num_fewshot 5 \
    --model_args "pretrained=dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1,max_new_tokens=256,steps=256,block_size=32,cfg_scale=0.0"

For BD3LM: use --model a2d_bd3lm and pretrained=dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1
"""

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import (
    BD3LMEvalHarness as BD3LMEvalHarnessBase,
    MDLMEvalConfig,
    MDLMEvalHarness,
)


@register_model("a2d_mdlm")
class A2DMDLMEvalHarness(MDLMEvalHarness):
    """A2D MDLM eval: thin subclass of core MDLMEvalHarness."""

    def __init__(self, config: MDLMEvalConfig | None = None, **kwargs):
        if config is None:
            config = MDLMEvalConfig()
        super().__init__(config=config, **kwargs)


@register_model("a2d_bd3lm")
class A2DBD3LMEvalHarness(BD3LMEvalHarnessBase):
    """A2D BD3LM eval: thin subclass of core BD3LMEvalHarness."""

    pass


if __name__ == "__main__":
    cli_evaluate()
