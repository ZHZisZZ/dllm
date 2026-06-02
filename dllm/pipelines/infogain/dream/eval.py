"""
accelerate launch --num_processes 4 \\
    dllm/pipelines/infogain/dream/eval.py \\
    --tasks gsm8k \\
    --model infogain_dream \\
    --apply_chat_template \\
    --num_fewshot 5 \\
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,max_new_tokens=256,steps=256,alg=info_gain,threshold=0.8,candidate_number=8,dtype=bfloat16,add_bos_token=True"
"""

from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.pipelines.dream.eval import DreamEvalConfig, DreamEvalHarness
from dllm.pipelines.infogain.dream import (
    InfoGainDreamConfig,
    InfoGainDreamSampler,
    InfoGainDreamSamplerConfig,
)


@dataclass
class InfoGainDreamEvalSamplerConfig(InfoGainDreamSamplerConfig):
    max_new_tokens: int = 128
    steps: int = 128
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    alg: str = "info_gain"


@dataclass
class InfoGainDreamEvalConfig(DreamEvalConfig):
    def get_model_config(self, pretrained: str):
        return InfoGainDreamConfig.from_pretrained(pretrained)


@register_model("infogain_dream")
class InfoGainDreamEvalHarness(DreamEvalHarness):
    def __init__(
        self,
        eval_config: InfoGainDreamEvalConfig | None = None,
        sampler_config: InfoGainDreamSamplerConfig | None = None,
        sampler_cls: type[InfoGainDreamSampler] = InfoGainDreamSampler,
        **kwargs,
    ) -> None:
        eval_config = eval_config or InfoGainDreamEvalConfig()
        sampler_config = sampler_config or InfoGainDreamEvalSamplerConfig()
        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
