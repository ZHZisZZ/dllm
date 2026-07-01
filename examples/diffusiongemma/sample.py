"""
python -u examples/diffusiongemma/sample.py \
    --model_name_or_path /mnt/weka/shrd/research/model/diffusiongemma-26B-A4B-it
"""

from dataclasses import dataclass
import importlib.util
import sys
import types
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion


def load_diffusiongemma_sampler():
    @dataclass
    class BaseSamplerOutput:
        sequences: torch.Tensor
        histories: list[torch.Tensor] | None = None

    @dataclass
    class BaseSamplerConfig:
        return_dict: bool = False

    @dataclass
    class BaseSampler:
        model: object
        tokenizer: object
        scheduler: object | None = None

    fake_base = types.ModuleType("dllm.core.samplers.base")
    fake_base.BaseSampler = BaseSampler
    fake_base.BaseSamplerConfig = BaseSamplerConfig
    fake_base.BaseSamplerOutput = BaseSamplerOutput
    sys.modules["dllm"] = types.ModuleType("dllm")
    sys.modules["dllm.core"] = types.ModuleType("dllm.core")
    sys.modules["dllm.core.samplers"] = types.ModuleType("dllm.core.samplers")
    sys.modules["dllm.core.samplers.base"] = fake_base

    repo_root = Path(__file__).resolve().parents[2]
    sampler_path = repo_root / "dllm/pipelines/diffusiongemma/sampler.py"
    spec = importlib.util.spec_from_file_location("diffusiongemma_sampler_example", sampler_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DiffusionGemmaSampler, module.DiffusionGemmaSamplerConfig


DiffusionGemmaSampler, DiffusionGemmaSamplerConfig = load_diffusiongemma_sampler()


@dataclass
class ScriptArguments:
    model_name_or_path: str = "/mnt/weka/shrd/research/model/diffusiongemma-26B-A4B-it"
    prompt: str | None = None
    prompt_file: str | None = None
    seed: int = 42
    dtype: str = "bfloat16"


@dataclass
class SamplerConfig(DiffusionGemmaSamplerConfig):
    max_new_tokens: int = 128
    block_size: int = 256
    max_denoising_steps: int = 48
    entropy_bound: float = 0.1
    entropy_threshold: float = 0.005
    stability_threshold: int = 1
    max_temperature: float = 0.8
    min_temperature: float = 0.4
    eos_token_id: int | None = None
    return_dict: bool = True


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

torch_dtype = getattr(torch, script_args.dtype)
processor = AutoProcessor.from_pretrained(script_args.model_name_or_path)
model = DiffusionGemmaForBlockDiffusion.from_pretrained(
    script_args.model_name_or_path,
    torch_dtype=torch_dtype,
    device_map="auto",
).eval()

sampler = DiffusionGemmaSampler(
    model=model,
    tokenizer=processor.tokenizer,
)

prompts = []
if script_args.prompt is not None:
    prompts.append(script_args.prompt)
if script_args.prompt_file is not None:
    prompts.extend(
        line.strip()
        for line in Path(script_args.prompt_file).read_text().splitlines()
        if line.strip()
    )
if not prompts:
    prompts = [
        "Give a concise explanation of text diffusion models.",
        "Write one short haiku about compilers.",
    ]

messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)

outputs = sampler.sample(inputs, sampler_config)
sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
decoded = processor.batch_decode(sequences, skip_special_tokens=True)

print("\n" + "=" * 80)
print("TEST: diffusiongemma.generate() via dllm sampler".center(80))
print("=" * 80)
for index, text in enumerate(decoded):
    print("\n" + "-" * 80)
    print(f"[Case {index}]")
    print("-" * 80)
    print(text.strip())
print("\n" + "=" * 80 + "\n")
