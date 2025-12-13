"""
Interactive chat / sampling script for LLaDA models.

Examples
--------
# Chat mode (multi-turn, chat template)
python -u examples/llada/chat.py --model_name_or_path "YOUR_MODEL_PATH"

# Raw single-turn sampling
python -u examples/llada/chat.py --model_name_or_path "YOUR_MODEL_PATH" --chat_template False
"""

import sys
from dataclasses import dataclass

import transformers

import dllm


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    seed: int = 42
    chat_template: bool = True
    visualize: bool = True

    def __post_init__(self):
        # same base-path resolution logic as in sample.py
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(dllm.core.samplers.MDLMSamplerConfig):
    steps: int = 128
    max_new_tokens: int = 128
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"


def main():
    parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

    model = dllm.utils.get_model(model_args=script_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
    sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)

    if script_args.chat_template:
        dllm.utils.multi_turn_chat(
            sampler=sampler,
            sampler_config=sampler_config,
            visualize=script_args.visualize,
        )
    else:
        print("\nSingle-turn sampling (no chat template).")
        dllm.utils.single_turn_sampling(
            sampler=sampler,
            sampler_config=sampler_config,
            visualize=script_args.visualize,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")
        sys.exit(0)
