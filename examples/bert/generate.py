"""
python -u examples/bert/generate.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from dataclasses import dataclass

import transformers

import dllm
from dllm.tools.chat import decode_trim
from dllm.pipelines import llada


@dataclass
class ScriptArguments:
    model_name_or_path: str = "dllm-collection/ModernBERT-large-chat-v0"
    seed: int = 42
    visualize: bool = True
    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )

@dataclass
class GeneratorConfig(llada.LLaDAGeneratorConfig):
    steps: int = 128
    max_new_tokens: int = 128
    block_length: int = 64
    temperature: float = 0.0
    remasking: str = "low_confidence"


parser = transformers.HfArgumentParser(
    (ScriptArguments, GeneratorConfig)
)
script_args, gen_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

# Load model & tokenizer
model = dllm.utils.get_model(model_args=script_args).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
generator = llada.LLaDAGenerator(model=model, tokenizer=tokenizer)
terminal_visualizer = dllm.core.generation.visualizer.TerminalVisualizer(tokenizer=tokenizer)

# --- Example 1: Batch generation ---
print("\n" + "=" * 80)
print("TEST: bert.generate()".center(80))
print("=" * 80)

messages = [
    [{"role": "user", "content": "Lily runs 12 km/h for 4 hours. How far in 8 hours?"}],
    # [{"role": "user", "content": "Please write an educational python function."}],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)
outputs = generator.generate(inputs, gen_config, return_dict_in_generate=True)
sequences = decode_trim(tokenizer, outputs.sequences.tolist(), inputs)

for iter, s in enumerate(sequences):
    print("\n" + "-" * 80)
    print(f"[Case {iter}]")
    print("-" * 80)
    print(s.strip() if s.strip() else "<empty>")
print("\n" + "=" * 80 + "\n")

if script_args.visualize: terminal_visualizer.visualize(outputs.histories, rich=True)
