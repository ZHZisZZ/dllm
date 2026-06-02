"""
Run from the dllm repo root:

    pytest scripts/tests/test_infogain_safety.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

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


class TinyNoCacheModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def forward(self, input_ids, *_, **__):
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
            device=input_ids.device,
        )
        logits[..., TinyTokenizer.mask_token_id] = -30.0
        return SimpleNamespace(logits=logits, past_key_values=None)


def test_resolve_generation_lengths_validates_bounds() -> None:
    assert ig.resolve_generation_lengths(4, None, 3) == (4, 7)
    assert ig.resolve_generation_lengths(4, 6, 3) == (3, 6)

    with pytest.raises(ValueError, match="Either max_new_tokens or max_length"):
        ig.resolve_generation_lengths(None, None, 3)
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        ig.resolve_generation_lengths(0, None, 3)
    with pytest.raises(ValueError, match="leave room"):
        ig.resolve_generation_lengths(None, 3, 3)


def test_info_gain_cost_reduction_supports_entropy_star() -> None:
    logits = torch.zeros(1, 4, 4)
    next_logits = torch.zeros(2, 4, 4)
    x_batch = torch.ones(2, 4, dtype=torch.long)
    actions = [torch.tensor([0, 1]), torch.tensor([2])]
    token_entropy = torch.log(torch.tensor(4.0))

    sum_cost, _, _ = ig.score_candidates(
        logits,
        next_logits,
        x_batch,
        actions,
        mask_id=0,
        cost_reduction="sum",
        future_weight=0.0,
    )
    mean_cost, _, weighted_j = ig.score_candidates(
        logits,
        next_logits,
        x_batch,
        actions,
        mask_id=0,
        cost_reduction="entropy*",
        cost_weight=2.0,
        future_weight=0.0,
    )

    assert torch.allclose(sum_cost, torch.stack([2 * token_entropy, token_entropy]))
    assert torch.allclose(mean_cost, torch.stack([token_entropy, token_entropy]))
    assert torch.allclose(weighted_j, -2.0 * mean_cost)

    with pytest.raises(ValueError, match="cost reduction"):
        ig.score_candidates(
            logits,
            next_logits,
            x_batch,
            actions,
            mask_id=0,
            cost_reduction="bad",
        )


def test_llada_info_gain_supports_batched_no_cache() -> None:
    sampler = InfoGainLLaDASampler(model=TinyNoCacheModel(), tokenizer=TinyTokenizer())
    cfg = InfoGainLLaDASamplerConfig(
        max_new_tokens=2,
        steps=2,
        block_size=1,
        use_cache=None,
        threshold=0.0,
        return_dict=True,
    )

    out = sampler.sample(
        [torch.tensor([1, 3]), torch.tensor([1, 4])],
        config=cfg,
    )

    assert out.sequences.shape == (2, 4)
    assert not (out.sequences[:, 2:] == TinyTokenizer.mask_token_id).any()
    assert len(out.histories) >= 2


def test_dream_info_gain_supports_batched_no_cache() -> None:
    sampler = InfoGainDreamSampler(model=TinyNoCacheModel(), tokenizer=TinyTokenizer())
    cfg = InfoGainDreamSamplerConfig(
        max_new_tokens=2,
        steps=2,
        block_size=1,
        use_cache=None,
        alg="info_gain",
        threshold=0.8,
        return_dict=True,
    )

    out = sampler.sample(
        [torch.tensor([1, 3]), torch.tensor([1, 4])],
        config=cfg,
    )

    assert out.sequences.shape == (2, 4)
    assert not (out.sequences[:, -2:] == TinyTokenizer.mask_token_id).any()
    assert len(out.histories) >= 2


def test_dream_confidence_threshold_supports_batched_no_cache() -> None:
    sampler = InfoGainDreamSampler(model=TinyNoCacheModel(), tokenizer=TinyTokenizer())
    cfg = InfoGainDreamSamplerConfig(
        max_new_tokens=3,
        steps=3,
        block_size=1,
        use_cache=None,
        alg="confidence_threshold",
        threshold=0.8,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        right_shift_logits=False,
        return_dict=True,
    )

    out = sampler.sample(
        [torch.tensor([1, 3]), torch.tensor([1, 4])],
        config=cfg,
    )

    assert out.sequences.shape == (2, 5)
    assert not (out.sequences[:, -3:] == TinyTokenizer.mask_token_id).any()


def test_dream_cache_requires_model_to_return_past_key_values() -> None:
    sampler = InfoGainDreamSampler(model=TinyNoCacheModel(), tokenizer=TinyTokenizer())
    cfg = InfoGainDreamSamplerConfig(
        max_new_tokens=2,
        steps=2,
        block_size=1,
        use_cache="prefix",
        alg="maskgit_plus",
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        right_shift_logits=False,
    )

    with pytest.raises(RuntimeError, match="past_key_values"):
        sampler.sample([torch.tensor([1, 3])], config=cfg)
