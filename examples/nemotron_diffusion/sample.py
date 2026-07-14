"""
python -u examples/nemotron_diffusion/sample.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from dataclasses import dataclass

import transformers

import dllm
from dllm.pipelines import nemotron_diffusion


@dataclass
class ScriptArguments:
    model_name_or_path: str = "nvidia/Nemotron-Labs-Diffusion-8B"
    dtype: str = "bfloat16"
    seed: int = 42

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(nemotron_diffusion.NemotronDiffusionSamplerConfig):
    max_new_tokens: int = 128
    # Matches the model's trained attention block size (config.json: block_size=32)
    # and the official README example; dllm's own default of 8 does not.
    block_length: int = 32
    threshold: float = 0.9
    temperature: float = 0.0
    # The official generate() example leaves this unset (no thinking-budget
    # enforcement); dllm's class default of 6000 would inject `</think>` and
    # diverge from that baseline, so pin it to None here to match.
    max_thinking_tokens: int | None = None


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

# Load model & tokenizer
# Nemotron-Labs-Diffusion ships its architecture as HF remote code and is only
# registered under `AutoModel` (see config.json's `auto_map`), so it is loaded
# directly instead of through `dllm.utils.get_model`/`get_tokenizer`.
model = transformers.AutoModel.from_pretrained(
    script_args.model_name_or_path,
    dtype=script_args.dtype,
    device_map="auto",
    trust_remote_code=True,
).eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(
    script_args.model_name_or_path, trust_remote_code=True
)
sampler = nemotron_diffusion.NemotronDiffusionSampler(model=model, tokenizer=tokenizer)

# --- Example: Batch sampling ---
print("\n" + "=" * 80)
print("TEST: nemotron_diffusion.sample()".center(80))
print("=" * 80)

messages = [
    [{"role": "user", "content": "Lily runs 12 km/h for 4 hours. How far in 8 hours?"}],
    [{"role": "user", "content": "Please write an educational python function."}],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)

outputs = sampler.sample(inputs, sampler_config, return_dict=True)
sequences = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)

for iter, s in enumerate(sequences):
    print("\n" + "-" * 80)
    print(f"[Case {iter}]")
    print("-" * 80)
    print(s.strip() if s.strip() else "<empty>")
print("\n" + "=" * 80 + "\n")
