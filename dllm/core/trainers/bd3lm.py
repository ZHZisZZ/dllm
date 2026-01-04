"""
References:

Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models:
https://arxiv.org/abs/2503.09573
"""

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from dllm.utils.collators import CollatorWrapper

from .mdlm import MDLMTrainer


@dataclass
class AppendEOSBlockWrapper(CollatorWrapper):
    block_size: int = 32

    def before(self, features):
        for ex in features:
            ids = ex["input_ids"]
            labs = ex["labels"]

            assert isinstance(ids, list) and isinstance(labs, list)

            L = len(ids)
            target = (L + self.block_size - 1) // self.block_size * self.block_size
            pad_len = target - L
            if pad_len > 0:
                ex["input_ids"] = ids + [self.tokenizer.eos_token_id] * pad_len
                ex["labels"] = labs + [self.tokenizer.eos_token_id] * pad_len
        return features


def _create_bd3lm_attention_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask for training
    composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        seq_len: Total sequence length.
        block_size: Defines the block structure.

    Returns:
        A boolean attention mask.
    """

    # Indicate whether token belongs to xt or x0
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    # Compute block indices
    block_q = torch.where(
        x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size
    )
    block_kv = torch.where(
        x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size
    )

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal


class BD3LMTrainer(MDLMTrainer):

    @dataclass
    class BD3LMConfig(MDLMTrainer.MDLMConfig):
        block_size: int = 32

    def __init__(
        self,
        args: BD3LMConfig,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)
        self.block_size = args.block_size

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: list[dict[str, Any]],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute the block diffusion language modeling loss.

        Applies block-wise diffusion with specialized attention masks where the model
        attends to both noised blocks (x_t) and clean context blocks (x_0).

        Args:
            model: The language model to train.
            inputs: Dictionary containing input_ids, labels, and optionally attention_mask.
            return_outputs: If True, return both loss and model outputs.

        Returns:
            Loss tensor, or tuple of (loss, outputs) if return_outputs is True.
        """
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape
        maskable_mask = labels != -100  # [b, l]

        # === 1. Sample diffusion timesteps ===
        # Each example draws a random timestep t ∈ [ε, 1), where ε avoids degenerate values near 0.
        # The scheduler defines the masking rate α(t); we convert it to a masking probability p_mask = 1 - α(t).
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )  # [b]
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)  # [b, l]

        # === 2. Apply stochastic masking ===
        # Tokens are masked independently according to p_mask(t).
        # Positions with label = -100 are excluded (ignored in loss).
        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        # Replace masked tokens with the special [MASK] token.
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        # === 3. Forward pass through the model (block-diffusion) ===
        # We follow the paper and feed x_t ⊕ x_0 with a specialized block mask.

        # concat_input_ids: [b, 2l], first l are noisy (x_t), last l are clean (x_0)
        concat_input_ids = torch.cat([noised_input_ids, input_ids], dim=1)

        # [TODO]: others like flash attention 2
        if self.accelerator.unwrap_model(model).config._attn_implementation == "sdpa":
            attention_mask = _create_bd3lm_attention_mask(
                b=None,
                h=None,
                q_idx=torch.arange(l * 2)[:, None],
                kv_idx=torch.arange(l * 2)[None, :],
                block_size=self.block_size,
                n=l,
            )
            attention_mask = (
                attention_mask.unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * l, 2 * l)
            )
            attention_mask = attention_mask.to(input_ids.device)
        elif (
            self.accelerator.unwrap_model(model).config._attn_implementation
            == "flex_attention"
        ):
            from torch.nn.attention.flex_attention import create_block_mask

            attention_mask = create_block_mask(
                partial(
                    _create_bd3lm_attention_mask, 
                    block_size=self.block_size, 
                    n=l
                ),
                B=None,
                H=None,
                Q_LEN=l * 2,
                KV_LEN=l * 2,
            )
        else:
            raise NotImplementedError

        base_pos = (
            torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        )  # [B, L]
        concat_position_ids = torch.cat([base_pos, base_pos], dim=1)  # [B, 2L]

        outputs = model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=concat_position_ids,
        )
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        logits = logits[:, :l]  # we only care about the first half for computing loss

        # === 4. Compute per-token loss weights ===
        # Depending on the configuration, weights may depend on timestep t
        # (e.g., scheduler-based) or be uniform (ones).
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_mask=masked_mask
        )

        # === 5. Compute weighted cross-entropy ===
        # Sanity check: ensure input_ids and labels match at valid positions
        assert (
            input_ids[maskable_mask] == labels[maskable_mask]
        ).all(), "Mismatch between input_ids and labels at valid positions"

        token_nll = F.cross_entropy(
            logits.transpose(1, 2),  # [b, V, l]
            input_ids,  # [b, l]
            reduction="none",  # [b, l]
        )
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)  # [b, l]

        self.meter.update(
            split="train" if model.training else "eval",
            value=token_nll.detach(),
            weight=maskable_mask.to(dtype=logits.dtype).detach(),
        )

        # === 6. Normalize loss ===
        if self.loss_norm_type == "token":
            token_nll /= maskable_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll /= maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b
        elif self.loss_norm_type == "batch":
            token_nll /= b
        else:
            raise ValueError("Invalid loss_norm_type.")
        loss = token_nll.sum()

        # === 7. Return final loss (and optionally model outputs) ===
        return (loss, outputs) if return_outputs else loss
