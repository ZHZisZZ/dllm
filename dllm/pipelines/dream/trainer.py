from typing import Any

import torch

from dllm.core.trainers import MDLMTrainer


def cart_weight(
    masked_indices: torch.Tensor, t: torch.Tensor, p: float = 0.3
) -> torch.Tensor:
    """
    Optimized CART weight computation using matrix operations.

    Args:
        masked_indices (torch.Tensor): (b, l) bool tensor indicating masked positions.
        t (torch.Tensor): (b,) time steps (0-1 sampled uniformly). Not directly used in CART.
        p (float): Parameter of geometric distribution (0 < p <= 1).

    Returns:
        torch.Tensor: (b, l) float tensor of weights.
    """
    b, l = masked_indices.shape
    device = masked_indices.device

    idx = torch.arange(l, device=device)
    dist_matrix = (idx[None, :] - idx[:, None]).abs() - 1
    dist_matrix = torch.clamp(dist_matrix, min=0)  # (l, l)
    geo_matrix = (
        torch.log(torch.tensor(p, device=device))
        + (dist_matrix - 1).clamp(min=0) * torch.log(torch.tensor(1 - p, device=device))
    ).exp() * 0.5  # Ensure numerical stability
    geo_matrix.masked_fill_(dist_matrix == 0, 0.0)  # ignore distance = 0

    valid_mask = (~masked_indices).float()  # (b, l), 1 = unmasked
    weights = valid_mask @ geo_matrix.T  # (b, l)
    weights = weights * masked_indices.float()
    return weights


class DreamTrainer(MDLMTrainer):
    """
    DreamTrainer: specialization of MDLMTrainer for Dream training.
    """

    def __init__(
        self,
        loss_weight_type: str = "cart[geo_p:0.3]",
        *args,
        **kwargs,
    ):
        super().__init__(
            loss_weight_type=loss_weight_type,
            *args,
            **kwargs,
        )

        self.right_shift_logits = True

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        inputs: dict[str, Any],
        masked_indices: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self.loss_weight_type.startswith("cart"):
            # parse geo_p
            import re

            match = re.search(r"geo_p:(0\.\d+)", self.loss_weight_type)
            geo_p = float(match.group(1)) if match else 0.3
            loss_weights = cart_weight(masked_indices, t, p=geo_p)
        else:
            loss_weights = super()._compute_loss_weights(
                t=t,
                inputs=inputs,
                masked_indices=masked_indices,
                *args,
                **kwargs,
            )
        return loss_weights
