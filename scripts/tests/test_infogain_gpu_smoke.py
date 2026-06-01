"""
Run from the dllm repo root on a GPU node:

    python scripts/tests/test_infogain_gpu_smoke.py

This smoke test avoids external model downloads. It verifies that the
Info-Gain LLaDA sampler runs end-to-end on CUDA and that candidate scoring is
batched into one model forward instead of N sequential forwards.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dllm.pipelines.infogain import info_gain_ops as ig
from dllm.pipelines.infogain.dream import (
    InfoGainDreamSampler,
    InfoGainDreamSamplerConfig,
)
from dllm.pipelines.infogain.llada import (
    InfoGainLLaDASampler,
    InfoGainLLaDASamplerConfig,
)


class TinyTokenizer:
    mask_token_id = 0
    bos_token_id = 1
    eos_token_id = 2


class TinyMaskedModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 256):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.mix = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
        )
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask=None, position_ids=None, **_):
        h = self.embedding(input_ids)
        # Non-causal context mixing: every position can depend on the current canvas.
        context = h.mean(dim=1, keepdim=True)
        h = self.mix(h + context)
        logits = self.lm_head(h)
        logits[..., TinyTokenizer.mask_token_id] = -30.0
        return SimpleNamespace(logits=logits)


@torch.no_grad()
def benchmark_candidate_scoring(device: torch.device) -> tuple[float, float, float]:
    torch.manual_seed(0)
    model = TinyMaskedModel(vocab_size=128, hidden_size=768).to(device).eval()
    bsz, seqlen, n_candidates = 1, 256, 16
    x = torch.randint(3, 128, (bsz, seqlen), device=device)
    x[:, 64:] = TinyTokenizer.mask_token_id
    mask_allowed = x == TinyTokenizer.mask_token_id

    logits = model(x).logits
    actions, x0s, _, _, _ = ig.generate_candidates(
        logits=logits,
        x=x,
        mask_allowed=mask_allowed,
        block_start=64,
        block_end=128,
        k=1,
        n_candidates=n_candidates,
        token_temp=0.0,
        pos_temp=0.2,
    )
    assert actions is not None and len(actions) > 1

    x_batch = x.expand(len(actions), -1).clone()
    for i, (act, x0) in enumerate(zip(actions, x0s)):
        x_batch[i, act] = x0[0, act]

    for _ in range(5):
        model(x_batch).logits
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        model(x_batch).logits
    if device.type == "cuda":
        torch.cuda.synchronize()
    batched = time.perf_counter() - t0

    for _ in range(5):
        for xb in x_batch:
            model(xb.unsqueeze(0)).logits
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        for xb in x_batch:
            model(xb.unsqueeze(0)).logits
    if device.type == "cuda":
        torch.cuda.synchronize()
    sequential = time.perf_counter() - t0

    return batched, sequential, sequential / batched


@torch.no_grad()
def smoke_decode_llada_batch(device: torch.device) -> torch.Tensor:
    torch.manual_seed(123)
    model = TinyMaskedModel().to(device).eval()
    sampler = InfoGainLLaDASampler(model=model, tokenizer=TinyTokenizer())
    cfg = InfoGainLLaDASamplerConfig(
        return_dict=True,
        max_new_tokens=16,
        steps=16,
        block_size=8,
        use_cache=None,
        threshold=0.0,
        candidate_number=8,
        position_temperature=0.2,
        variant="info_gain",
    )
    prompt = torch.tensor([[1, 7, 8, 9], [1, 4, 5, 6]], device=device)
    out = sampler.sample(prompt, config=cfg)
    assert out.sequences.device.type == device.type
    assert out.sequences.shape == (2, 20)
    assert not (out.sequences == TinyTokenizer.mask_token_id).any()
    assert len(out.histories) >= 2
    return out.sequences


@torch.no_grad()
def smoke_decode_dream_batch(device: torch.device) -> torch.Tensor:
    torch.manual_seed(456)
    model = TinyMaskedModel().to(device).eval()
    sampler = InfoGainDreamSampler(model=model, tokenizer=TinyTokenizer())
    cfg = InfoGainDreamSamplerConfig(
        return_dict=True,
        max_new_tokens=8,
        steps=8,
        block_size=4,
        use_cache=None,
        alg="info_gain",
        threshold=0.8,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        right_shift_logits=False,
        candidate_number=4,
        position_temperature=0.2,
        info_gain_variant="info_gain",
    )
    prompts = [
        torch.tensor([1, 7, 8], device=device),
        torch.tensor([1, 4, 5, 6], device=device),
    ]
    out = sampler.sample(prompts, config=cfg)
    assert out.sequences.device.type == device.type
    assert out.sequences.shape == (2, 12)
    assert not (out.sequences[:, -8:] == TinyTokenizer.mask_token_id).any()
    assert len(out.histories) >= 2
    return out.sequences


def main() -> None:
    assert torch.cuda.is_available(), "This test must run on a real GPU node."
    device = torch.device("cuda")
    llada_seq = smoke_decode_llada_batch(device)
    dream_seq = smoke_decode_dream_batch(device)
    batched, sequential, speedup = benchmark_candidate_scoring(device)
    print("INFOGAIN_GPU_SMOKE=PASS")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(
        "llada_batch_shape={} llada_tail0={} llada_tail1={}".format(
            tuple(llada_seq.shape),
            llada_seq[0, -8:].tolist(),
            llada_seq[1, -8:].tolist(),
        )
    )
    print(
        "dream_batch_shape={} dream_tail0={} dream_tail1={}".format(
            tuple(dream_seq.shape),
            dream_seq[0, -8:].tolist(),
            dream_seq[1, -8:].tolist(),
        )
    )
    print(f"batched_candidate_seconds={batched:.4f}")
    print(f"sequential_candidate_seconds={sequential:.4f}")
    print(f"candidate_batch_speedup={speedup:.2f}x")
    assert speedup > 1.15, f"Expected batched candidate scoring speedup, got {speedup:.2f}x"


if __name__ == "__main__":
    main()
