"""
References:

Simple and Effective Masked Diffusion Language Models:
https://arxiv.org/abs/2406.07524

Large Language Diffusion Models:
https://arxiv.org/abs/2502.09992
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler
from dllm.utils.data import prepend_bos
from .utils import EpochPPLMeter


class MDLMTrainer(transformers.Trainer):

    def __init__(
        self,
        scheduler: BaseAlphaScheduler | None = None,
        time_epsilon: float = 1e-3,
        loss_weight_type: str = "scheduler",  # "scheduler", "uniform"
        loss_normalization_type: str = "sequence",  # "batch", "sequence", "token"
        right_shift_logits: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not (0.0 < time_epsilon < 1.0):
            raise ValueError("time_epsilon must be in (0, 1)")

        self.scheduler = scheduler or LinearAlphaScheduler()
        self.time_epsilon = time_epsilon
        self.loss_weight_type = loss_weight_type
        self.loss_normalization_type = loss_normalization_type
        self.right_shift_logits = right_shift_logits

        self.epoch_meter = EpochPPLMeter(self, train_prefix="train", eval_prefix="eval")
        self.add_callback(self.epoch_meter)

    def _preprocess_inputs(self, inputs):
        if self.right_shift_logits:
            labels = inputs.get("labels", None)

            # If labels exist and EVERY sequence already starts with -100,
            # we treat them as is and skip prepending BOS.
            if labels is not None:
                # shape: [bsz, seq_len]
                if torch.all(labels[:, 0] == -100):
                    return inputs

            # Otherwise, prepend BOS (and corresponding labels / attention_mask).
            inputs = prepend_bos(
                inputs,
                bos_token_id=self.processing_class.bos_token_id,
                label_pad_token_id=-100,
            )
        return inputs

    def _postprocess_outputs(self, outputs):
        if self.right_shift_logits:
            logits = outputs.logits
            outputs.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return outputs

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        inputs: dict[str, Any],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute loss weights given timestep t and other arguments."""
        b, l = inputs["input_ids"].shape
        if self.loss_weight_type == "scheduler":
            loss_weights = self.scheduler.weight(t).unsqueeze(1).repeat(1, l)
        elif self.loss_weight_type == "uniform":
            loss_weights = torch.ones_like(inputs["input_ids"])
        else:
            raise NotImplementedError
        return loss_weights

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()

        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().contiguous()

        return (loss.detach(), logits, labels)

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute the masked diffusion language modeling loss.

        Applies stochastic masking to input tokens based on a diffusion timestep,
        then computes the weighted cross-entropy loss for predicting the original tokens.

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
        token_cnt_per_seq = torch.sum(labels != -100, dim=1, keepdim=True)  # [b, 1]

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
        masked_indices = (torch.rand((b, l), device=input_ids.device) < p_mask) & (
            labels != -100
        )
        # Replace masked tokens with the special [MASK] token.
        noised_input_ids = torch.where(
            masked_indices, self.processing_class.mask_token_id, input_ids
        )

        # === 3. Forward pass through the model ===
        # The model predicts clean tokens given noised inputs.
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        # === 4. Handle degenerate cases (no tokens masked) ===
        # If no positions were masked, return a zero loss to keep gradients valid.
        # This step is necessary for Deepspeed Zero-{2,3}
        if not masked_indices.any():
            self.epoch_meter.update(
                split="train" if model.training else "eval", 
                nll_sum=logits.sum() * 0.0, 
                token_cnt=token_cnt_per_seq.sum(),
            )
            return (
                (logits.sum() * 0.0, outputs) if return_outputs else logits.sum() * 0.0
            )

        # === 5. Compute per-token loss weights ===
        # Depending on the configuration, weights may depend on timestep t
        # (e.g., scheduler-based) or be uniform (ones).
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_indices=masked_indices
        )

        # === 6. Compute weighted cross-entropy ===
        # Only masked tokens contribute to the loss.
        assert (input_ids[masked_indices] == labels[masked_indices]).all()
        token_loss = F.cross_entropy(
            logits[masked_indices], input_ids[masked_indices], reduction="none"
        )
        token_loss = token_loss * loss_weights[masked_indices]

        # === 7. Normalize loss ===
        if self.loss_normalization_type == "batch":
            token_loss_normalized = token_loss / b
        elif self.loss_normalization_type == "sequence":
            token_loss_normalized = token_loss / token_cnt_per_seq.expand(-1, l)[masked_indices] / b
        elif self.loss_normalization_type == "token":
            token_loss_normalized = token_loss / token_cnt_per_seq.sum()
        else:
            raise ValueError("Invalid loss_normalization_type.")
        loss = token_loss_normalized.sum()

        # === 8. Return final loss (and optionally model outputs) ===
        self.epoch_meter.update(
            split="train" if model.training else "eval", 
            nll_sum=token_loss.sum(), 
            token_cnt=token_cnt_per_seq.sum(),
        ) # `nll_sum / token_cnt` is equivalent to `loss` when `self.loss_normalization_type == "token"``
        return (loss, outputs) if return_outputs else loss
