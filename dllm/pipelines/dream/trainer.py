from typing import Any
from dataclasses import dataclass

import torch

from dllm.core.trainers import MDLMTrainer


def cart_weight(
    masked_mask: torch.Tensor, t: torch.Tensor, p: float = 0.3
) -> torch.Tensor:
    """
    Optimized CART weight computation using matrix operations.

    Args:
        masked_mask (torch.Tensor): (b, l) bool tensor indicating masked positions.
        t (torch.Tensor): (b,) time steps (0-1 sampled uniformly). Not directly used in CART.
        p (float): Parameter of geometric distribution (0 < p <= 1).

    Returns:
        torch.Tensor: (b, l) float tensor of weights.
    """
    b, l = masked_mask.shape
    device = masked_mask.device

    idx = torch.arange(l, device=device)
    dist_matrix = (idx[None, :] - idx[:, None]).abs() - 1
    dist_matrix = torch.clamp(dist_matrix, min=0)  # (l, l)
    geo_matrix = (
        torch.log(torch.tensor(p, device=device))
        + (dist_matrix - 1).clamp(min=0) * torch.log(torch.tensor(1 - p, device=device))
    ).exp() * 0.5  # Ensure numerical stability
    geo_matrix.masked_fill_(dist_matrix == 0, 0.0)  # ignore distance = 0

    valid_mask = (~masked_mask).float()  # (b, l), 1 = unmasked
    weights = valid_mask @ geo_matrix.T  # (b, l)
    weights = weights * masked_mask.float()
    return weights


class DreamTrainer(MDLMTrainer):
    """
    DreamTrainer: specialization of MDLMTrainer for Dream training.
    """

    @dataclass
    class DreamConfig(MDLMTrainer.MDLMConfig):
        loss_weight_type: str = "cart[geo_p:0.3]"
        right_shift_logits: bool = True

        def __post_init__(self):
            super().__post_init__()
            assert self.right_shift_logits

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        inputs: dict[str, Any],
        masked_mask: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self.loss_weight_type.startswith("cart"):
            # parse geo_p
            import re

            match = re.search(r"geo_p:(0\.\d+)", self.loss_weight_type)
            geo_p = float(match.group(1)) if match else 0.3
            loss_weights = cart_weight(masked_mask, t, p=geo_p)
        else:
            loss_weights = super()._compute_loss_weights(
                t=t,
                inputs=inputs,
                masked_mask=masked_mask,
                *args,
                **kwargs,
            )
        return loss_weights
