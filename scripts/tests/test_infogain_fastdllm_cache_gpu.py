"""
Run from the dllm repo root on a GPU node:

    python scripts/tests/test_infogain_fastdllm_cache_gpu.py
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from dllm.pipelines.fastdllm.llada import FastdLLMLLaDASampler, FastdLLMLLaDASamplerConfig
from dllm.pipelines.fastdllm.dream import FastdLLMDreamSampler, FastdLLMDreamSamplerConfig
from dllm.pipelines.infogain.llada import InfoGainLLaDASampler, InfoGainLLaDASamplerConfig
from dllm.pipelines.infogain.dream import InfoGainDreamSampler, InfoGainDreamSamplerConfig


class TinyTokenizer:
    mask_token_id = 0
    bos_token_id = 1
    eos_token_id = 2


class CacheAwareTinyModel(torch.nn.Module):
    """Small deterministic model that exposes the cache contract Fast-dLLM needs."""

    def __init__(self, vocab_size: int = 64, hidden_size: int = 32):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.calls: list[dict[str, object]] = []

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        replace_position: torch.Tensor | None = None,
        **_: object,
    ):
        self.calls.append(
            {
                "seq_len": int(input_ids.shape[1]),
                "use_cache": bool(use_cache),
                "has_past": past_key_values is not None,
                "has_replace": replace_position is not None,
                "replace_count": int(replace_position.sum().item()) if replace_position is not None else 0,
            }
        )

        hidden = torch.tanh(self.embedding(input_ids))
        logits = self.proj(hidden)
        position_bias = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=logits.dtype)
        logits = logits + position_bias.view(1, -1, 1) * 0.01
        logits[..., TinyTokenizer.mask_token_id] = -30.0

        pkv = None
        if use_cache:
            batch, seq_len = input_ids.shape
            old_len = 0 if past_key_values is None else int(past_key_values[0][0].shape[2])
            total_len = old_len + seq_len
            dummy = torch.zeros(batch, 1, total_len, 1, device=input_ids.device, dtype=logits.dtype)
            pkv = [(dummy, dummy)]

        return SimpleNamespace(logits=logits, past_key_values=pkv)


class CacheAwareTinyDreamModel(CacheAwareTinyModel):
    """Dream cache tensors use [batch, seq, dim] instead of LLaDA's [batch, heads, seq, dim]."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        replace_position: torch.Tensor | None = None,
        **_: object,
    ):
        self.calls.append(
            {
                "seq_len": int(input_ids.shape[1]),
                "use_cache": bool(use_cache),
                "has_past": past_key_values is not None,
                "has_replace": replace_position is not None,
                "replace_count": int(replace_position.sum().item()) if replace_position is not None else 0,
            }
        )

        hidden = torch.tanh(self.embedding(input_ids))
        logits = self.proj(hidden)
        position_bias = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=logits.dtype)
        logits = logits + position_bias.view(1, -1, 1) * 0.01
        logits[..., TinyTokenizer.mask_token_id] = -30.0

        pkv = None
        if use_cache:
            batch, seq_len = input_ids.shape
            old_len = 0 if past_key_values is None else int(past_key_values[0][0].shape[1])
            total_len = old_len + seq_len
            dummy = torch.zeros(batch, total_len, 1, device=input_ids.device, dtype=logits.dtype)
            pkv = [(dummy, dummy)]

        return SimpleNamespace(logits=logits, past_key_values=pkv)


def run_one_llada_mode(mode: str, device: torch.device) -> dict[str, object]:
    torch.manual_seed(7)
    base_model = CacheAwareTinyModel()
    fast_model = copy.deepcopy(base_model).to(device).eval()
    info_model = copy.deepcopy(base_model).to(device).eval()

    tokenizer = TinyTokenizer()
    prompt = torch.tensor([[1, 5, 6, 7]], device=device)

    common = dict(
        max_new_tokens=8,
        steps=8,
        block_size=4,
        temperature=0.0,
        remasking="low_confidence",
        threshold=None,
        use_cache=mode,
        return_dict=True,
    )
    fast_cfg = FastdLLMLLaDASamplerConfig(**common)
    info_cfg = InfoGainLLaDASamplerConfig(**common)

    with torch.no_grad():
        fast_out = FastdLLMLLaDASampler(fast_model, tokenizer).sample(prompt, config=fast_cfg)
        info_out = InfoGainLLaDASampler(info_model, tokenizer).sample(prompt, config=info_cfg)

    same = torch.equal(fast_out.sequences, info_out.sequences)
    if not same:
        raise AssertionError(f"{mode}: Info-Gain cache delegation changed Fast-dLLM output")
    if torch.any(info_out.sequences[:, prompt.shape[1] :] == tokenizer.mask_token_id):
        raise AssertionError(f"{mode}: generated suffix still contains mask tokens")

    info_saw_past = any(call["has_past"] for call in info_model.calls)
    info_saw_replace = any(call["has_replace"] for call in info_model.calls)
    if not info_saw_past:
        raise AssertionError(f"{mode}: cache path never reused past_key_values")
    if mode == "dual" and not info_saw_replace:
        raise AssertionError("dual: cache path never passed replace_position")
    if mode == "prefix" and info_saw_replace:
        raise AssertionError("prefix: replace_position should not be used")

    return {
        "mode": mode,
        "same": same,
        "fast_calls": len(fast_model.calls),
        "info_calls": len(info_model.calls),
        "info_saw_past": info_saw_past,
        "info_saw_replace": info_saw_replace,
        "tail": info_out.sequences[0, -8:].detach().cpu().tolist(),
    }


def run_one_dream_mode(mode: str, device: torch.device) -> dict[str, object]:
    torch.manual_seed(11)
    base_model = CacheAwareTinyDreamModel()
    fast_model = copy.deepcopy(base_model).to(device).eval()
    info_model = copy.deepcopy(base_model).to(device).eval()

    tokenizer = TinyTokenizer()
    prompt = [torch.tensor([1, 5, 6, 7], device=device)]

    common = dict(
        max_new_tokens=8,
        steps=8,
        block_size=4,
        alg="maskgit_plus",
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        threshold=None,
        right_shift_logits=False,
        use_cache=mode,
        return_dict=True,
    )
    fast_cfg = FastdLLMDreamSamplerConfig(**common)
    info_cfg = InfoGainDreamSamplerConfig(**common)

    with torch.no_grad():
        fast_out = FastdLLMDreamSampler(fast_model, tokenizer).sample(prompt, config=fast_cfg)
        info_out = InfoGainDreamSampler(info_model, tokenizer).sample(prompt, config=info_cfg)

    same = torch.equal(fast_out.sequences, info_out.sequences)
    if not same:
        raise AssertionError(f"dream/{mode}: Info-Gain cache path changed Fast-dLLM output")
    if torch.any(info_out.sequences[:, -8:] == tokenizer.mask_token_id):
        raise AssertionError(f"dream/{mode}: generated suffix still contains mask tokens")

    info_saw_past = any(call["has_past"] for call in info_model.calls)
    info_saw_replace = any(call["has_replace"] for call in info_model.calls)
    if not info_saw_past:
        raise AssertionError(f"dream/{mode}: cache path never reused past_key_values")
    if mode == "dual" and not info_saw_replace:
        raise AssertionError("dream/dual: cache path never passed replace_position")
    if mode == "prefix" and info_saw_replace:
        raise AssertionError("dream/prefix: replace_position should not be used")

    return {
        "mode": f"dream/{mode}",
        "same": same,
        "fast_calls": len(fast_model.calls),
        "info_calls": len(info_model.calls),
        "info_saw_past": info_saw_past,
        "info_saw_replace": info_saw_replace,
        "tail": info_out.sequences[0, -8:].detach().cpu().tolist(),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU cache smoke test")
    device = torch.device("cuda")
    print("torch", torch.__version__, torch.cuda.get_device_name(0))
    for mode in ("prefix", "dual"):
        result = run_one_llada_mode(mode, device)
        print(
            "mode=llada/{mode} same={same} fast_calls={fast_calls} info_calls={info_calls} "
            "info_saw_past={info_saw_past} info_saw_replace={info_saw_replace} tail={tail}".format(**result)
        )
    for mode in ("prefix", "dual"):
        result = run_one_dream_mode(mode, device)
        print(
            "mode={mode} same={same} fast_calls={fast_calls} info_calls={info_calls} "
            "info_saw_past={info_saw_past} info_saw_replace={info_saw_replace} tail={tail}".format(**result)
        )
    print("INFOGAIN_FASTDLLM_CACHE_GPU_SMOKE=PASS")


if __name__ == "__main__":
    main()
