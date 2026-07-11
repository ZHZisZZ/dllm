"""
DiffusionGemma trainer utilities.

Example:
    python -m pytest scripts/tests/test_diffusiongemma_trainer.py -q
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from dllm.utils.configs import TrainingArguments


@dataclass
class DiffusionGemmaTrainerConfig(TrainingArguments):
    canvas_length: int = 256
    time_epsilon: float = 1e-4
    self_conditioning_prob: float = 0.5
    loss_norm_type: str = "token"
    train_on_prompt: bool = False

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        if self.canvas_length <= 0:
            raise ValueError("canvas_length must be positive.")
        if not 0.0 < self.time_epsilon < 1.0:
            raise ValueError("time_epsilon must be in (0, 1).")
        if not 0.0 <= self.self_conditioning_prob <= 1.0:
            raise ValueError("self_conditioning_prob must be in [0, 1].")
        if self.loss_norm_type not in {"token", "sequence", "batch"}:
            raise ValueError("loss_norm_type must be one of: token, sequence, batch.")


class DiffusionGemmaTrainer(transformers.Trainer):
    def __init__(
        self,
        args: DiffusionGemmaTrainerConfig,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)

        self.canvas_length = args.canvas_length
        self.time_epsilon = args.time_epsilon
        self.self_conditioning_prob = args.self_conditioning_prob
        self.loss_norm_type = args.loss_norm_type
        self.train_on_prompt = args.train_on_prompt
        self.label_names = getattr(self, "label_names", None) or ["labels"]

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        input_ids = inputs["input_ids"]
        labels = inputs.get("labels", input_ids)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if self.train_on_prompt:
            labels = torch.where(attention_mask, input_ids, torch.full_like(input_ids, -100))

        batch = self._prepare_batch(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            model=model,
        )

        self_conditioning_logits = None
        if self.self_conditioning_prob > 0.0:
            with torch.no_grad():
                self_conditioning_logits = model(
                    input_ids=batch.context_ids,
                    attention_mask=batch.context_attention_mask,
                    decoder_input_ids=batch.decoder_input_ids,
                    decoder_attention_mask=batch.decoder_attention_mask,
                ).logits.detach()
            if self.self_conditioning_prob < 1.0:
                do_self_conditioning = (
                    torch.rand((batch.context_ids.shape[0],), device=batch.context_ids.device)
                    < self.self_conditioning_prob
                )
                self_conditioning_logits = torch.where(
                    do_self_conditioning[:, None, None],
                    self_conditioning_logits,
                    torch.zeros_like(self_conditioning_logits),
                )

        outputs = model(
            input_ids=batch.context_ids,
            attention_mask=batch.context_attention_mask,
            decoder_input_ids=batch.decoder_input_ids,
            decoder_attention_mask=batch.decoder_attention_mask,
            self_conditioning_logits=self_conditioning_logits,
        )
        logits = outputs.logits

        token_nll = F.cross_entropy(
            logits.transpose(1, 2),
            batch.target_ids,
            reduction="none",
        )
        token_nll = token_nll * batch.target_mask.to(token_nll.dtype)
        if self.loss_norm_type == "token":
            loss = token_nll.sum() / batch.target_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            per_sequence = token_nll.sum(dim=-1) / batch.target_mask.sum(dim=-1).clamp_min(1)
            loss = per_sequence.mean()
        else:
            loss = token_nll.sum() / token_nll.shape[0]

        return (loss, outputs) if return_outputs else loss

    def _prepare_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        model: transformers.PreTrainedModel | nn.Module,
    ) -> SimpleNamespace:
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        text_config = model.config.text_config
        pad_id = 0 if text_config.pad_token_id is None else int(text_config.pad_token_id)
        vocab_size = int(text_config.vocab_size)
        target_mask = (labels != -100) & attention_mask

        block_start = torch.empty(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            valid_positions = target_mask[i].nonzero(as_tuple=False).flatten()
            valid_positions = valid_positions[valid_positions > 0]
            if valid_positions.numel() == 0:
                raise ValueError("No trainable token with non-empty context in sample.")

            start = int(valid_positions[0].item())
            starts = torch.arange(start, seq_len, self.canvas_length, device=device)
            valid = starts[target_mask[i, starts]]
            if valid.numel() == 0:
                valid = valid_positions

            choice = torch.randint(valid.numel(), (1,), device=device)
            block_start[i] = valid[choice]

        context_len = int(block_start.max().item())
        if context_len <= 0:
            raise ValueError("DiffusionGemma training requires at least one context token per sample.")

        context_shape = (batch_size, context_len)
        canvas_shape = (batch_size, self.canvas_length)
        context_ids = input_ids.new_full(context_shape, pad_id)
        target_ids = input_ids.new_full(canvas_shape, pad_id)
        context_attention_mask = attention_mask.new_zeros(context_shape)
        canvas_attention_mask = attention_mask.new_zeros(canvas_shape)
        target_mask_canvas = target_mask.new_zeros(canvas_shape)

        for i in range(batch_size):
            start = int(block_start[i].item())
            context_ids[i, :start] = input_ids[i, :start]
            context_attention_mask[i, :start] = attention_mask[i, :start]

            end = min(start + self.canvas_length, seq_len)
            width = max(end - start, 0)
            if width > 0:
                target_ids[i, :width] = input_ids[i, start:end]
                canvas_attention_mask[i, :width] = attention_mask[i, start:end]
                target_mask_canvas[i, :width] = target_mask[i, start:end]

        if not target_mask_canvas.any():
            raise ValueError("No trainable canvas tokens found in this batch.")

        noise_proportion = self.time_epsilon + (1.0 - self.time_epsilon) * torch.rand(
            (batch_size,),
            device=device,
        )
        noise_mask = torch.rand(
            (batch_size, self.canvas_length),
            device=device,
        ) < noise_proportion[:, None]
        noise_mask = noise_mask & target_mask_canvas
        needs_noise = target_mask_canvas.any(dim=-1) & ~noise_mask.any(dim=-1)
        for i in needs_noise.nonzero(as_tuple=False).flatten():
            valid = target_mask_canvas[i].nonzero(as_tuple=False).flatten()
            choice = torch.randint(valid.numel(), (1,), device=device)
            noise_mask[i, valid[choice]] = True
        random_tokens = torch.randint(
            low=0,
            high=vocab_size,
            size=target_ids.shape,
            dtype=torch.long,
            device=device,
        )
        decoder_input_ids = torch.where(noise_mask, random_tokens, target_ids)
        decoder_attention_mask = torch.cat([context_attention_mask, canvas_attention_mask], dim=1)

        return SimpleNamespace(
            context_ids=context_ids,
            context_attention_mask=context_attention_mask,
            target_ids=target_ids,
            canvas_attention_mask=canvas_attention_mask,
            target_mask=target_mask_canvas & noise_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            block_start=block_start,
            noise_mask=noise_mask,
            noise_proportion=noise_proportion,
        )
