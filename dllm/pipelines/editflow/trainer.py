from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from dllm.core.schedulers import BaseKappaScheduler, CubicKappaScheduler
from dllm.pipelines.editflow.utils import pad_1d
from dllm.utils.configs import TrainingArguments

BLANK = -1


def align_with_blanks(
    x0: List[int], x1: List[int], sub_cost: int = 1, gap_cost: int = 1
) -> Dict:
    """
    Needleman–Wunsch global alignment of two integer sequences with:
        match cost = 0, substitution cost = sub_cost, gap cost = gap_cost.
    Returns aligned sequences (z0, z1) of equal length containing BLANK = ε where gaps occur.
    """
    n, m = len(x0), len(x1)
    # DP tables
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ptr = [[None] * (m + 1) for _ in range(n + 1)]  # 'diag', 'up', 'left'

    for i in range(1, n + 1):
        dp[i][0] = i * gap_cost
        ptr[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = j * gap_cost
        ptr[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_diag = dp[i - 1][j - 1] + (0 if x0[i - 1] == x1[j - 1] else sub_cost)
            cost_up = dp[i - 1][j] + gap_cost
            cost_left = dp[i][j - 1] + gap_cost
            best = min(cost_diag, cost_up, cost_left)
            dp[i][j] = best
            if best == cost_diag:
                ptr[i][j] = "diag"
            elif best == cost_up:
                ptr[i][j] = "up"
            else:
                ptr[i][j] = "left"

    # traceback
    z0, z1 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        p = ptr[i][j]
        if p == "diag":
            z0.append(x0[i - 1])
            z1.append(x1[j - 1])
            i -= 1
            j -= 1
        elif p == "up":
            z0.append(x0[i - 1])
            z1.append(BLANK)
            i -= 1
        else:  # 'left'
            z0.append(BLANK)
            z1.append(x1[j - 1])
            j -= 1
    z0.reverse()
    z1.reverse()
    # return Alignment(z0=z0, z1=z1)
    # return {"z0": z0, "z1": z1}
    return dict(z0=z0, z1=z1)


# def align_with_blanks(
#     x0: list[int], x1: list[int], sub_cost: int = 1, gap_cost: int = 1
# ) -> dict:
#     """
#     Needleman–Wunsch with a secondary objective that defers gaps to the end:
#       - 'up' (gap in z1) is penalized if j < m
#       - 'left' (gap in z0) is penalized if i < n
#     This pushes blanks (-1) to the *right* whether x0 > x1 or x0 < x1.
#     """
#     n, m = len(x0), len(x1)

#     dp_cost = [[0] * (m + 1) for _ in range(n + 1)]
#     dp_pen = [[0] * (m + 1) for _ in range(n + 1)]
#     ptr = [[None] * (m + 1) for _ in range(n + 1)]  # 'diag' | 'up' | 'left'

#     # Left edge: all 'up' moves with j=0 (< m) → penalize each step
#     for i in range(1, n + 1):
#         dp_cost[i][0] = i * gap_cost
#         dp_pen[i][0] = i  # i early 'up' moves
#         ptr[i][0] = "up"

#     # Top edge: all 'left' moves with i=0 (< n) → penalize each step
#     for j in range(1, m + 1):
#         dp_cost[0][j] = j * gap_cost
#         dp_pen[0][j] = j  # j early 'left' moves
#         ptr[0][j] = "left"

#     for i in range(1, n + 1):
#         xi = x0[i - 1]
#         for j in range(1, m + 1):
#             yj = x1[j - 1]

#             # diag
#             cost_diag = dp_cost[i - 1][j - 1] + (0 if xi == yj else sub_cost)
#             pen_diag = dp_pen[i - 1][j - 1]
#             cand_diag = (cost_diag, pen_diag)

#             # up: add blank to z1, penalize if j < m (early)
#             cost_up = dp_cost[i - 1][j] + gap_cost
#             pen_up = dp_pen[i - 1][j] + (1 if j < m else 0)
#             cand_up = (cost_up, pen_up)

#             # left: add blank to z0, penalize if i < n (early)
#             cost_left = dp_cost[i][j - 1] + gap_cost
#             pen_left = dp_pen[i][j - 1] + (1 if i < n else 0)
#             cand_left = (cost_left, pen_left)

#             # choose (cost,pen) min; deterministic tie-break: diag > left > up
#             best = min(cand_diag, cand_left, cand_up)
#             dp_cost[i][j], dp_pen[i][j] = best
#             if best == cand_diag:
#                 ptr[i][j] = "diag"
#             elif best == cand_left:
#                 ptr[i][j] = "left"
#             else:
#                 ptr[i][j] = "up"

#     # traceback
#     z0, z1 = [], []
#     i, j = n, m
#     while i > 0 or j > 0:
#         p = ptr[i][j]
#         if p == "diag":
#             z0.append(x0[i - 1])
#             z1.append(x1[j - 1])
#             i -= 1
#             j -= 1
#         elif p == "up":
#             z0.append(x0[i - 1])
#             z1.append(BLANK)
#             i -= 1
#         else:  # 'left'
#             z0.append(BLANK)
#             z1.append(x1[j - 1])
#             j -= 1

#     z0.reverse()
#     z1.reverse()
#     return dict(z0=z0, z1=z1)


def strip_blanks(z: list[int]) -> list[int]:
    # IMPORTANT: do NOT strip BOS; we only remove BLANKs
    return [t for t in z if t != BLANK]


@dataclass
class Edit:
    kind: str  # "SUB" | "DEL" | "INS"
    pos: int  # position (for SUB/DEL) or token-row idx for INS (incl. BOS row 0)
    token: int | None  # token for SUB/INS, else None


def build_remaining_edits(zt: list[int], z1: list[int]) -> list[Edit]:
    edits: list[Edit] = []

    def count_nonblank_prefix(z: list[int], j: int) -> int:
        c = 0
        for k in range(j):
            if z[k] != BLANK:
                c += 1
        return c

    for j, (a, b) in enumerate(zip(zt, z1)):
        if a == b:
            continue
        nb = count_nonblank_prefix(
            zt, j
        )  # counts BOS as 1, first content token will be nb=1 before its column

        if a == BLANK and b != BLANK:
            # INSERT after row (nb-1): BOS insert => nb=1 -> gap=0; general case works too
            gap = max(nb - 1, 0)
            edits.append(Edit("INS", gap, b))

        elif a != BLANK and b == BLANK:
            # DELETE token at row nb (first content token => nb=1, allowed; BOS is never BLANK so nb>=1)
            pos = nb
            # if pos > 0:   # forbid BOS (row 0)
            edits.append(Edit("DEL", pos, None))

        else:  # a != BLANK, b != BLANK, a != b
            # SUB token at row nb
            pos = nb
            # if pos > 0:   # forbid BOS (row 0)
            edits.append(Edit("SUB", pos, b))
    return edits


class EditFlowTrainer(transformers.Trainer):
    """
    Trainer for Edit Flows where the model returns:
      - sub_logits: [B,L,V]   (token dist for SUB)
      - ins_logits: [B,L,V]   (token dist for INS)
      - sub_rate_hat: [B,L]   (normalized rates; NO kappa factor)
      - del_rate_hat: [B,L]
      - ins_rate_hat: [B,L]
    True intensities are  w * rate_hat, with w = kappa_dot(t) / (1 - kappa(t)).
    """

    @dataclass
    class EditFlowConfig(TrainingArguments):
        time_epsilon: float = 1e-3
        normalize_per_position: bool = True
        max_w: float = 20.0

    def __init__(
        self,
        args: EditFlowConfig,
        scheduler: BaseKappaScheduler | None = None,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)

        self.scheduler = scheduler if scheduler is not None else CubicKappaScheduler()
        self.time_epsilon = args.time_epsilon
        self.normalize_per_position = args.normalize_per_position
        self.max_w = args.max_w

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        device = self.model.device
        B = len(inputs["x0_ids"])

        # -------- 1) Align with blanks (z0,z1) and sample time t --------
        aligns = [
            align_with_blanks(x0, x1)
            for x0, x1 in zip(inputs["x0_ids"], inputs["x1_ids"])
        ]
        z0_list = [a["z0"] for a in aligns]
        z1_list = [a["z1"] for a in aligns]
        assert all(len(z0) == len(z1) for z0, z1 in zip(z0_list, z1_list))
        assert all(z0[0] != BLANK for z0 in z0_list)  # BOS must remain

        t = (1 - self.time_epsilon) * torch.rand(B, 1, device=device)  # [B,1]
        k = self.scheduler.kappa(t).to(device)  # [B,1]
        w = self.scheduler.weight(t).squeeze(1).to(device)  # [B]
        if self.max_w:
            w = w.clamp(max=self.max_w)

        # -------- 2) Sample z_t by κ-mixing (vectorized per example) --------
        # Keep python lists -> tensors per-example to reuse build_remaining_edits
        zt_list: list[list[int]] = []
        for z0, z1, kb in zip(z0_list, z1_list, k.squeeze(1).tolist()):
            # per-column Bernoulli(κ) mix; BOS is equal in z0/z1 so it stays BOS
            choose_target = torch.rand(len(z0)) < kb
            zt = [b if choose_target[j] else a for j, (a, b) in enumerate(zip(z0, z1))]
            zt_list.append(zt)

        # -------- 3) Strip blanks to x_t and compute remaining edits --------
        xt_list = [strip_blanks(zt) for zt in zt_list]
        edits_list: list[list[Edit]] = [
            build_remaining_edits(zt, z1) for zt, z1 in zip(zt_list, z1_list)
        ]

        # -------- 4) Collate x_t for the model --------
        x_tok, x_mask = pad_1d(
            xt_list, pad_val=self.processing_class.pad_token_id
        )  # [B,Lmax], [B,Lmax]
        x_tok = x_tok.to(device)
        x_mask = x_mask.to(device)

        # -------- 5) Forward pass --------
        out = model(input_ids=x_tok, attention_mask=x_mask, t=t.to(device))
        # Rename for clarity: model returns normalized rates (no kappa)
        sub_rate_hat = out["sub_rate_hat"]  # [B,L]
        del_rate_hat = out["del_rate_hat"]  # [B,L]
        ins_rate_hat = out["ins_rate_hat"]  # [B,L]
        logQ_sub = F.log_softmax(out["sub_logits"], dim=-1)  # [B,L,V]
        logQ_ins = F.log_softmax(out["ins_logits"], dim=-1)  # [B,L,V]

        # *** NEW: zero-cost anchor to "touch" every head even if unused this step ***
        # Using .sum() * 0.0 keeps a graph dependency without changing the loss value.
        # Include both logits (for SUB/INS heads) and rates (for SUB/DEL/INS heads).
        # This is important for Deepspeed ZeRO stage 2/3 to avoid skipping unused parameters.
        anchor = (
            sub_rate_hat.sum() * 0.0
            + del_rate_hat.sum() * 0.0
            + ins_rate_hat.sum() * 0.0
            + logQ_sub.sum() * 0.0
            + logQ_ins.sum() * 0.0
        )

        # Utility
        def safe_log(x: torch.Tensor) -> torch.Tensor:
            return torch.log(x.clamp_min(1e-12))

        # -------- 6) Survival term --------
        # Survival = E[sum of true intensities over valid rows]
        # true intensity = w[b] * rate_hat[b, i]
        mask_f = x_mask.float()
        # L = mask_f.sum(dim=1).clamp_min(1.0)           # [B] number of positions (incl. BOS)
        L1 = torch.tensor(
            [len(x1) for x1 in inputs["x1_ids"]], device=device, dtype=torch.float
        ).clamp_min(1.0)
        denom = L1 if self.normalize_per_position else torch.ones_like(L1)

        Lambda_hat = ((sub_rate_hat + del_rate_hat + ins_rate_hat) * mask_f).sum(
            dim=1
        )  # [B]
        loss_surv = ((w * Lambda_hat) / denom).mean()

        # -------- 7) Positive edit terms --------
        # For each remaining edit e:  -log true rate(e)  - log token prob(e) if tokenized
        # loss_pos_per = sub_rate_hat.new_zeros(B)  # [B]
        # for b, edits in enumerate(edits_list):
        #     if not edits:
        #         continue
        #     cur_len = int(x_mask[b].sum().item())
        #     for e in edits:
        #         pos = e.pos
        #         assert 0 <= pos < cur_len, f"pos {pos} out of range {cur_len}"
        #         if e.kind == "SUB":
        #             loss_pos_per[b] -= logQ_sub[b, pos, e.token] + safe_log(
        #                 sub_rate_hat[b, pos]
        #             )
        #         elif e.kind == "DEL":
        #             loss_pos_per[b] -= safe_log(del_rate_hat[b, pos])
        #         else:  # "INS"
        #             loss_pos_per[b] -= logQ_ins[b, pos, e.token] + safe_log(
        #                 ins_rate_hat[b, pos]
        #             )

        # -------- 7) Positive edit terms (vectorized) --------
        pos_sub, tok_sub, pos_ins, tok_ins, pos_del = [], [], [], [], []
        for b, edits in enumerate(edits_list):
            cur_len = int(x_mask[b].sum().item())
            ps, ts, pi, ti, pd = [], [], [], [], []
            for e in edits:
                if not (0 <= e.pos < cur_len):
                    raise AssertionError(
                        f"pos {e.pos} out of range {cur_len} for b={b}"
                    )
                if e.kind == "SUB":
                    ps.append(e.pos)
                    ts.append(e.token)
                elif e.kind == "INS":
                    pi.append(e.pos)
                    ti.append(e.token)
                else:
                    pd.append(e.pos)
            pos_sub.append(
                torch.tensor(ps, device=x_tok.device, dtype=torch.long) if ps else None
            )
            tok_sub.append(
                torch.tensor(ts, device=x_tok.device, dtype=torch.long) if ts else None
            )
            pos_ins.append(
                torch.tensor(pi, device=x_tok.device, dtype=torch.long) if pi else None
            )
            tok_ins.append(
                torch.tensor(ti, device=x_tok.device, dtype=torch.long) if ti else None
            )
            pos_del.append(
                torch.tensor(pd, device=x_tok.device, dtype=torch.long) if pd else None
            )

        loss_pos_terms = []
        for b in range(B):
            lp = x_tok.new_zeros(())
            if pos_sub[b] is not None:
                lp = (
                    lp
                    - (
                        logQ_sub[b, pos_sub[b], tok_sub[b]]
                        + safe_log(sub_rate_hat[b, pos_sub[b]])
                    ).sum()
                )
            if pos_ins[b] is not None:
                lp = (
                    lp
                    - (
                        logQ_ins[b, pos_ins[b], tok_ins[b]]
                        + safe_log(ins_rate_hat[b, pos_ins[b]])
                    ).sum()
                )
            if pos_del[b] is not None:
                lp = lp - safe_log(del_rate_hat[b, pos_del[b]]).sum()
            loss_pos_terms.append(lp)
        loss_pos_per = torch.stack(loss_pos_terms)  # [B]

        # # Average positive term per sequence (MC estimator across batch)
        loss_pos = ((w * loss_pos_per) / denom).mean()

        # -------- 8) Total --------
        loss = loss_surv + loss_pos + anchor
        return (loss, out) if return_outputs else loss


if __name__ == "__main__":
    pass
