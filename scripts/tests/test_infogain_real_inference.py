from __future__ import annotations

import argparse
import re
import time
from dataclasses import asdict

import torch
import transformers

import dllm


DEFAULT_PROMPT = "Lily runs 12 km/h for 4 hours. How far will Lily run in 8 hours?"
DEFAULT_EXPECTED_RE = r"\b(96|ninety[- ]?six)\b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-model Info-Gain inference correctness smoke test.")
    parser.add_argument("--pipeline", choices=["llada", "dream"], required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--expected_regex", default=DEFAULT_EXPECTED_RE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--candidate_number", type=int, default=8)
    parser.add_argument("--position_temperature", type=float, default=0.2)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["none"],
        help="Cache modes to test. Use 'none' for the Info-Gain objective; prefix/dual test Fast-dLLM cache branches.",
    )
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def make_inputs(tokenizer, prompt: str):
    messages = [[{"role": "user", "content": prompt}]]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )


def trim_outputs(tokenizer, outputs, inputs) -> list[str]:
    trimmed = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)
    return [text.strip() for text in trimmed]


def check_generation(text: str, expected_regex: str, mode: str) -> None:
    if not text:
        raise AssertionError(f"{mode}: empty decoded output")
    if re.search(r"<\|?mask\|?>|\[MASK\]", text, flags=re.IGNORECASE):
        raise AssertionError(f"{mode}: decoded output still contains a mask token: {text!r}")
    if not re.search(expected_regex, text, flags=re.IGNORECASE):
        raise AssertionError(
            f"{mode}: decoded output did not match {expected_regex!r}.\nOutput:\n{text}"
        )


def run_llada(args: argparse.Namespace) -> None:
    cfg = dllm.pipelines.infogain.llada.InfoGainLLaDAConfig.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )
    model_args = argparse.Namespace(model_name_or_path=args.model_name_or_path)
    model = dllm.utils.get_model(model_args=model_args, config=cfg).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    sampler = dllm.pipelines.infogain.llada.InfoGainLLaDASampler(model=model, tokenizer=tokenizer)
    inputs = make_inputs(tokenizer, args.prompt)

    for mode in args.modes:
        sampler_config = dllm.pipelines.infogain.llada.InfoGainLLaDASamplerConfig(
            steps=args.steps,
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            temperature=0.0,
            remasking="low_confidence",
            use_cache=mode,
            threshold=args.threshold if mode == "none" else None,
            candidate_number=args.candidate_number,
            position_temperature=args.position_temperature,
            variant="info_gain",
            return_dict=True,
        )
        start = time.time()
        with torch.no_grad():
            outputs = sampler.sample(inputs, config=sampler_config, return_dict=True)
        elapsed = time.time() - start
        texts = trim_outputs(tokenizer, outputs, inputs)
        for text in texts:
            check_generation(text, args.expected_regex, f"llada/{mode}")
        print(f"mode=llada/{mode} seconds={elapsed:.2f} config={asdict(sampler_config)}")
        print("decoded=" + texts[0].replace("\n", "\\n"))


def run_dream(args: argparse.Namespace) -> None:
    cfg = dllm.pipelines.infogain.dream.InfoGainDreamConfig.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )
    model_args = argparse.Namespace(model_name_or_path=args.model_name_or_path)
    model = dllm.utils.get_model(model_args=model_args, config=cfg).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    sampler = dllm.pipelines.infogain.dream.InfoGainDreamSampler(model=model, tokenizer=tokenizer)
    inputs = make_inputs(tokenizer, args.prompt)

    for mode in args.modes:
        sampler_config = dllm.pipelines.infogain.dream.InfoGainDreamSamplerConfig(
            steps=args.steps,
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            temperature=0.0,
            top_p=None,
            top_k=None,
            alg="info_gain" if mode == "none" else "maskgit_plus",
            threshold=args.threshold if mode == "none" else None,
            use_cache=mode,
            candidate_number=args.candidate_number,
            position_temperature=args.position_temperature,
            info_gain_variant="info_gain",
            return_dict=True,
        )
        start = time.time()
        with torch.no_grad():
            outputs = sampler.sample(inputs, sampler_config, return_dict=True)
        elapsed = time.time() - start
        texts = trim_outputs(tokenizer, outputs, inputs)
        for text in texts:
            check_generation(text, args.expected_regex, f"dream/{mode}")
        print(f"mode=dream/{mode} seconds={elapsed:.2f} config={asdict(sampler_config)}")
        print("decoded=" + texts[0].replace("\n", "\\n"))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real-model inference correctness")
    transformers.set_seed(args.seed)
    resolved = dllm.utils.resolve_with_base_env(args.model_name_or_path, "BASE_MODELS_DIR")
    args.model_name_or_path = resolved
    print("torch", torch.__version__, torch.cuda.get_device_name(0))
    print("pipeline", args.pipeline)
    print("model", args.model_name_or_path)
    print("prompt", args.prompt)
    print("expected_regex", args.expected_regex)
    if args.pipeline == "llada":
        run_llada(args)
    else:
        run_dream(args)
    print("INFOGAIN_REAL_INFERENCE_CORRECTNESS=PASS")


if __name__ == "__main__":
    main()
