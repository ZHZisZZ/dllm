"""
python -u examples/infogain/dream/sample.py --model_name_or_path "Dream-org/Dream-v0-Instruct-7B"
"""

import time
from dataclasses import dataclass

import transformers

import dllm


@dataclass
class ScriptArguments:
    model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
    seed: int = 42
    visualize: bool = True

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(dllm.pipelines.infogain.dream.InfoGainDreamSamplerConfig):
    steps: int = 256
    max_new_tokens: int = 256
    block_size: int = 32
    temperature: float = 0.0
    top_p: float = None
    top_k: int = None
    alg: str = "info_gain"
    threshold: float = 0.8
    use_cache: str = "none"
    candidate_number: int = 8
    position_temperature: float = 0.2
    info_gain_variant: str = "info_gain"


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)
dream_cfg = dllm.pipelines.infogain.dream.InfoGainDreamConfig.from_pretrained(
    script_args.model_name_or_path
)

model = dllm.utils.get_model(model_args=script_args, config=dream_cfg).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
sampler = dllm.pipelines.infogain.dream.InfoGainDreamSampler(model=model, tokenizer=tokenizer)
terminal_visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)

print("\n" + "=" * 80)
print("TEST: infogain dream.sample()".center(80))
print("=" * 80)

messages = [
    [{"role": "user", "content": "Lily runs 12 km/h for 4 hours. How far in 8 hours?"}],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)

start = time.time()
outputs = sampler.sample(inputs, sampler_config, return_dict=True)
end = time.time()
sequences = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)

for i, s in enumerate(sequences):
    print("\n" + "-" * 80)
    print(f"[Case {i}]")
    print("-" * 80)
    print(s.strip() if s.strip() else "<empty>")
print("\n" + "=" * 80 + "\n")

if script_args.visualize:
    terminal_visualizer.visualize(outputs.histories, rich=True)

print(
    f"Config: use_cache={sampler_config.use_cache}, alg={sampler_config.alg}, "
    f"threshold={sampler_config.threshold}, candidate_number={sampler_config.candidate_number}"
)
print(
    f"Total NFE:{len(outputs.histories) - 1}. Time taken for sampling: {end - start:.2f} seconds"
)
print(
    f"Token speed: {(len(outputs.sequences[0])-len(inputs[0]))*1.0/(end - start):.2f} tokens/s"
)
