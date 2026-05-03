"""
python -u examples/infogain/llada/sample.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct"
"""

import time
from dataclasses import dataclass

import transformers

import dllm


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    seed: int = 42
    visualize: bool = True

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(dllm.pipelines.infogain.llada.InfoGainLLaDASamplerConfig):
    steps: int = 256
    max_new_tokens: int = 256
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
    use_cache: str = "none"
    threshold: float = 0.8
    candidate_number: int = 8
    position_temperature: float = 0.2
    variant: str = "info_gain"
    begin_suppress_tokens: list[int] = None


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)
cfg = dllm.pipelines.infogain.llada.InfoGainLLaDAConfig.from_pretrained(
    script_args.model_name_or_path
)

model = dllm.utils.get_model(model_args=script_args, config=cfg).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
sampler = dllm.pipelines.infogain.llada.InfoGainLLaDASampler(model=model, tokenizer=tokenizer)
terminal_visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)

print("\n" + "=" * 80)
print("TEST: infogain llada.sample()".center(80))
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
outputs = sampler.sample(inputs, config=sampler_config, return_dict=True)
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
    f"Config: use_cache={sampler_config.use_cache}, threshold={sampler_config.threshold}, "
    f"candidate_number={sampler_config.candidate_number}, variant={sampler_config.variant}"
)
print(
    f"Total NFE:{len(outputs.histories) - 1}. Time taken for sampling: {end - start:.2f} seconds"
)
print(
    f"Token speed: {(len(outputs.sequences[0])-len(inputs[0]))*1.0/(end - start):.2f} tokens/s"
)
