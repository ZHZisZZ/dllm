"""
Shared Info-Gain scoring utilities (ported from Information-Gain-Sampler).

Reference: https://github.com/yks23/Information-Gain-Sampler
Paper: Yang et al., "Improving Sampling for Masked Diffusion Models via Information Gain" (arXiv:2602.18176).

Run smoke test: python -c "import torch; from dllm.pipelines.infogain import info_gain_ops as ig; print(ig.compute_entropy(torch.randn(1,3,5)).shape)"
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-position Shannon entropy: [B, L, V] → [B, L]."""
    p = F.softmax(logits.float(), dim=-1).clamp(min=eps)
    return -(p * p.log()).sum(-1)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits
    u = torch.zeros_like(logits).uniform_().clamp(1e-10, 1 - 1e-10)
    gumbel = -(-u.log()).log()
    return logits + temperature * gumbel


def generate_candidates(
    logits: torch.Tensor,
    x: torch.Tensor,
    mask_allowed: torch.Tensor,
    block_start: int,
    block_end: int,
    k: int,
    n_candidates: int,
    token_temp: float,
    pos_temp: float,
):
    """
    Generate N diverse candidate actions via Gumbel sampling.

    Returns (actions, x0s, conf_base, valid, probs_base) or (None, x0_base, ...) for trivial path.
    """
    device = x.device
    neg = torch.finfo(torch.float32).min

    block_mask = torch.zeros_like(mask_allowed)
    block_mask[:, block_start:block_end] = mask_allowed[:, block_start:block_end]

    x0_base = torch.argmax(add_gumbel_noise(logits, token_temp), dim=-1)
    x0_base = torch.where(mask_allowed, x0_base, x)

    probs_base = F.softmax(logits.float(), dim=-1)
    conf_base = torch.gather(probs_base, -1, x0_base.unsqueeze(-1)).squeeze(-1)
    conf_base = torch.where(block_mask, conf_base, torch.full_like(conf_base, neg))

    valid = torch.where(conf_base[0] > neg)[0]
    nv = valid.shape[0]

    if nv == 0 or nv <= k or pos_temp <= 0 or n_candidates <= 1:
        return None, x0_base, conf_base, valid, probs_base

    actions, x0s, seen = [], [], set()
    for c in range(n_candidates):
        if c == 0:
            x0_c, conf_c = x0_base, conf_base
        else:
            x0_c = torch.argmax(add_gumbel_noise(logits, token_temp), dim=-1)
            x0_c = torch.where(mask_allowed, x0_c, x)
            cf = torch.gather(probs_base, -1, x0_c.unsqueeze(-1)).squeeze(-1)
            conf_c = torch.where(block_mask, cf, torch.full_like(cf, neg))

        vc = conf_c[0, valid]
        if c == 0:
            _, tk = torch.topk(vc, min(k, nv))
        else:
            g = -torch.log(-torch.log(torch.rand(nv, device=device) + 1e-10) + 1e-10)
            _, tk = torch.topk(vc / pos_temp + g, min(k, nv))

        act = valid[tk]
        key = tuple(sorted(act.tolist()))
        if key not in seen:
            seen.add(key)
            actions.append(act)
            x0s.append(x0_c)

    return actions, x0s, conf_base, valid, probs_base


def score_candidates(
    logits: torch.Tensor,
    next_logits: torch.Tensor,
    x_batch: torch.Tensor,
    actions: list,
    mask_id: int,
    variant: str = "info_gain",
):
    """Return (C, H_next, J) per candidate; higher J is better."""
    ne = compute_entropy(next_logits)
    rm = x_batch == mask_id
    H_next = torch.where(rm, ne, ne.new_zeros(1)).sum(-1) / (rm.sum(-1).float() + 1e-10)

    if variant == "lookum":
        J = -H_next
        C = H_next.new_zeros(H_next.shape)
    else:
        ce = compute_entropy(logits)
        C = torch.stack([ce[0, a].sum() for a in actions])
        J = -C - H_next

    return C, H_next, J


def greedy_unmask_block(
    x: torch.Tensor,
    logits: torch.Tensor,
    mask_allowed: torch.Tensor,
    bs: int,
    be: int,
    temperature: float,
    mask_id: int,
):
    """Fill all remaining masks in [bs, be) greedily (high-confidence bypass)."""
    x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)
    x0 = torch.where(mask_allowed, x0, x)
    out = x.clone()
    m = mask_allowed.clone()
    m[:, :bs] = False
    m[:, be:] = False
    out = torch.where(m, x0, out)
    return out
