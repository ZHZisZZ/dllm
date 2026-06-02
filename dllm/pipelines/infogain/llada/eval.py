"""
accelerate launch --num_processes 4 \\
    dllm/pipelines/infogain/llada/eval.py \\
    --tasks gsm8k \\
    --model infogain_llada \\
    --apply_chat_template \\
    --num_fewshot 5 \\
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,use_cache=none,max_new_tokens=256,steps=256,block_size=32,threshold=0.8,candidate_number=8,suppress_tokens=[],begin_suppress_tokens=[]"
"""

from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness
from dllm.pipelines.infogain.llada import (
    InfoGainLLaDAConfig,
    InfoGainLLaDASampler,
    InfoGainLLaDASamplerConfig,
)


@dataclass
class InfoGainLLaDAEvalSamplerConfig(InfoGainLLaDASamplerConfig):
    max_new_tokens: int = 1024
    steps: int = 1024
    block_size: int = 1024


@dataclass
class InfoGainLLaDAEvalConfig(MDLMEvalConfig):
    max_length: int = 4096

    def get_model_config(self, pretrained: str):
        return InfoGainLLaDAConfig.from_pretrained(pretrained)


@register_model("infogain_llada")
class InfoGainLLaDAEvalHarness(MDLMEvalHarness):
    def __init__(
        self,
        eval_config: InfoGainLLaDAEvalConfig | None = None,
        sampler_config: InfoGainLLaDASamplerConfig | None = None,
        sampler_cls: type[InfoGainLLaDASampler] = InfoGainLLaDASampler,
        **kwargs,
    ):
        eval_config = eval_config or InfoGainLLaDAEvalConfig()
        sampler_config = sampler_config or InfoGainLLaDAEvalSamplerConfig()
        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
