"""
python -u examples/andi/mdlm/sample.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from dataclasses import dataclass

import torch
import transformers

import dllm


@dataclass
class ScriptArguments:
    model_name_or_path: str = "models/andi/Qwen3-0.6B/ar+mdlm/uniform/alpaca/checkpoint-final"
    seed: int = 42
    visualize: bool = True

    def __post_init__(self):
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
    right_shift_logits: bool = False


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

# Load model & tokenizer
model = dllm.utils.get_model(model_args=script_args).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)
terminal_visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)


messages = [
    [{"role": "user", "content": "Lily runs 12 km/h for 4 hours. How far in 8 hours?"}],
    [{"role": "user", "content": "Please write an educational python function."}],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)


# --- Example 1: Diff sampling ---
print("\n" + "=" * 80)
print("TEST: sampler.sample()".center(80))
print("=" * 80)

# you can just use `model.mode = "diff"`
with model.mode_ctx("diff"):
    outputs = sampler.sample(inputs, sampler_config, return_dict=True)
sequences = dllm.utils.decode_trim(tokenizer, outputs.sequences.tolist(), inputs)

for iter, s in enumerate(sequences):
    print("\n" + "-" * 80)
    print(f"[Case {iter}]")
    print("-" * 80)
    print(s.strip() if s.strip() else "<empty>")
print("\n" + "=" * 80 + "\n")


# --- Example 2: AR sampling ---
print("\n" + "=" * 80)
print("TEST: model.generate()".center(80))
print("=" * 80)

# you can just use `model.mode = "ar"`
with model.mode_ctx("ar"):
    # inputs: List[List[int]]
    assert isinstance(inputs, list) and inputs and isinstance(inputs[0], list)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_len = max(len(x) for x in inputs)

    # left padding
    input_ids = torch.tensor(
        [[pad_id] * (max_len - len(x)) + x for x in inputs],
        dtype=torch.long,
        device=model.device,
    )
    attention_mask = (input_ids != pad_id).long()

    gen_kwargs = dict(
        max_new_tokens=sampler_config.max_new_tokens,
        do_sample=bool(sampler_config.temperature and sampler_config.temperature > 0),
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = float(sampler_config.temperature)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
# trim and decode
sequences = dllm.utils.decode_trim(tokenizer, output_ids.tolist(), inputs if not isinstance(inputs, dict) else inputs["input_ids"])

for iter, s in enumerate(sequences):
    print("\n" + "-" * 80)
    print(f"[Case {iter}]")
    print("-" * 80)
    print(s.strip() if s.strip() else "<empty>")
print("\n" + "=" * 80 + "\n")
