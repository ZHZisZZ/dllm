"""
DiffusionGemma trainer utilities.

Example:
    python -m pytest scripts/tests/test_diffusiongemma_trainer.py -q
"""

from __future__ import annotations

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


class DiffusionGemmaTrainer(transformers.Trainer):
    def __init__(
        self,
        args: DiffusionGemmaTrainerConfig,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)

        if args.canvas_length <= 0:
            raise ValueError("canvas_length must be positive.")
        if not 0.0 < args.time_epsilon < 1.0:
            raise ValueError("time_epsilon must be in (0, 1).")
        if not 0.0 <= args.self_conditioning_prob <= 1.0:
            raise ValueError("self_conditioning_prob must be in [0, 1].")
        if args.loss_norm_type not in {"token", "sequence", "batch"}:
            raise ValueError("loss_norm_type must be one of: token, sequence, batch.")

        self.canvas_length = args.canvas_length
        self.time_epsilon = args.time_epsilon
        self.self_conditioning_prob = args.self_conditioning_prob
        self.loss_norm_type = args.loss_norm_type
        self.train_on_prompt = args.train_on_prompt

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
        del kwargs

        input_ids = inputs.get("input_ids")
        if input_ids is None:
            input_ids = inputs["prompt"]
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
            prompt_len=inputs.get("prompt_len"),
            model=model,
            inputs=inputs,
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
        loss = self._normalize_loss(token_nll, batch.target_mask)

        return (loss, outputs) if return_outputs else loss

    def _prepare_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: torch.Tensor | list[int] | None,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
    ) -> SimpleNamespace:
        if "canvas" in inputs and "prompt" in inputs:
            return self._prepare_canvas_batch(inputs=inputs, model=model)

        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        pad_id = self._pad_token_id(model)
        vocab_size = self._vocab_size(model)
        target_mask = (labels != -100) & attention_mask

        if prompt_len is not None and not isinstance(prompt_len, torch.Tensor):
            prompt_len = torch.as_tensor(prompt_len, dtype=torch.long, device=device)
        elif isinstance(prompt_len, torch.Tensor):
            prompt_len = prompt_len.to(device=device, dtype=torch.long)

        block_start = self._sample_block_start(
            target_mask=target_mask,
            prompt_len=prompt_len,
        )
        context_len = int(block_start.max().item())
        if context_len <= 0:
            raise ValueError("DiffusionGemma training requires at least one context token per sample.")

        context_ids = torch.full(
            (batch_size, context_len),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        context_attention_mask = torch.zeros(
            (batch_size, context_len),
            dtype=torch.bool,
            device=device,
        )
        target_ids = torch.full(
            (batch_size, self.canvas_length),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        canvas_attention_mask = torch.zeros(
            (batch_size, self.canvas_length),
            dtype=torch.bool,
            device=device,
        )
        target_mask_canvas = torch.zeros(
            (batch_size, self.canvas_length),
            dtype=torch.bool,
            device=device,
        )

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

        decoder_input_ids, noise_mask, noise_proportion = self._corrupt_canvas_tokens(
            target_ids=target_ids,
            target_mask=target_mask_canvas,
            vocab_size=vocab_size,
        )
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

    def _prepare_canvas_batch(
        self,
        inputs: dict[str, torch.Tensor | Any],
        model: transformers.PreTrainedModel | nn.Module,
    ) -> SimpleNamespace:
        prompt = inputs["prompt"]
        canvas = inputs["canvas"]
        if canvas.ndim == 3 and canvas.shape[-1] == 1:
            canvas = canvas.squeeze(-1)

        device = prompt.device
        batch_size, total_canvas_len = canvas.shape
        if total_canvas_len % self.canvas_length != 0:
            raise ValueError("canvas length must be divisible by trainer canvas_length.")

        pad_id = self._pad_token_id(model)
        vocab_size = self._vocab_size(model)
        prompt = prompt.to(device=device, dtype=torch.long)
        canvas = canvas.to(device=device, dtype=torch.long)
        prompt_attention_mask = inputs.get("prompt_mask")
        if prompt_attention_mask is None:
            prompt_attention_mask = prompt != pad_id
        else:
            prompt_attention_mask = prompt_attention_mask.to(device=device).bool()

        canvas_mask = inputs.get("canvas_mask")
        if canvas_mask is None:
            canvas_mask = canvas != pad_id
        else:
            canvas_mask = canvas_mask.to(device=device).bool()

        canvas_id = inputs.get("canvas_id")
        num_canvases = total_canvas_len // self.canvas_length
        if canvas_id is None:
            canvas_id = torch.arange(num_canvases, device=device).repeat_interleave(self.canvas_length)
            canvas_id = canvas_id.unsqueeze(0).expand(batch_size, -1)
        else:
            canvas_id = canvas_id.to(device=device, dtype=torch.long)

        selected_canvas_idx = self._sample_canvas_index(canvas_mask=canvas_mask, num_canvases=num_canvases)
        offsets = selected_canvas_idx * self.canvas_length
        position = torch.arange(self.canvas_length, device=device).unsqueeze(0) + offsets.unsqueeze(1)

        target_ids = torch.gather(canvas, dim=1, index=position)
        canvas_attention_mask = torch.gather(canvas_mask, dim=1, index=position)
        selected_ids = torch.gather(canvas_id, dim=1, index=position)
        target_mask = canvas_attention_mask & (selected_ids == selected_canvas_idx[:, None])
        if not target_mask.any():
            raise ValueError("No trainable canvas tokens found in this batch.")

        context_ids = torch.full_like(prompt, pad_id)
        context_attention_mask = prompt_attention_mask.clone()
        context_ids[context_attention_mask] = prompt[context_attention_mask]

        decoder_input_ids, noise_mask, noise_proportion = self._corrupt_canvas_tokens(
            target_ids=target_ids,
            target_mask=target_mask,
            vocab_size=vocab_size,
        )
        decoder_attention_mask = torch.cat([context_attention_mask, canvas_attention_mask], dim=1)

        return SimpleNamespace(
            context_ids=context_ids,
            context_attention_mask=context_attention_mask,
            target_ids=target_ids,
            canvas_attention_mask=canvas_attention_mask,
            target_mask=target_mask & noise_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            block_start=offsets,
            selected_canvas_idx=selected_canvas_idx,
            noise_mask=noise_mask,
            noise_proportion=noise_proportion,
        )

    def _sample_block_start(
        self,
        target_mask: torch.Tensor,
        prompt_len: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len = target_mask.shape
        device = target_mask.device
        block_start = torch.empty(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            if prompt_len is not None:
                start = int(prompt_len[i].item())
                if start <= 0:
                    start = 1
                starts = torch.arange(start, seq_len, self.canvas_length, device=device)
                valid = starts[target_mask[i, starts]]
                if valid.numel() == 0:
                    valid_positions = target_mask[i].nonzero(as_tuple=False).flatten()
                    valid_positions = valid_positions[valid_positions > 0]
                    if valid_positions.numel() == 0:
                        raise ValueError("No trainable token with non-empty context in sample.")
                    valid = valid_positions
            else:
                valid = target_mask[i].nonzero(as_tuple=False).flatten()
                valid = valid[valid > 0]
                if valid.numel() == 0:
                    raise ValueError("No trainable token with non-empty context in sample.")

            choice = torch.randint(valid.numel(), (1,), device=device)
            block_start[i] = valid[choice]

        return block_start

    def _corrupt_canvas_tokens(
        self,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        vocab_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, canvas_length = target_ids.shape
        noise_proportion = self.time_epsilon + (1.0 - self.time_epsilon) * torch.rand(
            (batch_size,),
            device=target_ids.device,
        )
        noise_mask = torch.rand(
            (batch_size, canvas_length),
            device=target_ids.device,
        ) < noise_proportion[:, None]
        noise_mask = noise_mask & target_mask
        noise_mask = self._ensure_noised_target(noise_mask=noise_mask, target_mask=target_mask)
        random_tokens = torch.randint(
            low=0,
            high=vocab_size,
            size=target_ids.shape,
            dtype=torch.long,
            device=target_ids.device,
        )
        decoder_input_ids = torch.where(noise_mask, random_tokens, target_ids)
        return decoder_input_ids, noise_mask, noise_proportion

    @staticmethod
    def _ensure_noised_target(noise_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        needs_noise = target_mask.any(dim=-1) & ~noise_mask.any(dim=-1)
        if not needs_noise.any():
            return noise_mask

        for i in needs_noise.nonzero(as_tuple=False).flatten():
            valid = target_mask[i].nonzero(as_tuple=False).flatten()
            choice = torch.randint(valid.numel(), (1,), device=target_mask.device)
            noise_mask[i, valid[choice]] = True
        return noise_mask

    def _sample_canvas_index(self, canvas_mask: torch.Tensor, num_canvases: int) -> torch.Tensor:
        batch_size = canvas_mask.shape[0]
        first_token_indices = torch.arange(num_canvases, device=canvas_mask.device) * self.canvas_length
        canvas_validity = canvas_mask[:, first_token_indices]
        selected_canvas_idx = torch.empty(batch_size, dtype=torch.long, device=canvas_mask.device)
        for i in range(batch_size):
            valid = canvas_validity[i].nonzero(as_tuple=False).flatten()
            if valid.numel() == 0:
                raise ValueError("No valid canvas found in sample.")
            choice = torch.randint(valid.numel(), (1,), device=canvas_mask.device)
            selected_canvas_idx[i] = valid[choice]
        return selected_canvas_idx

    def _normalize_loss(self, token_nll: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        if self.loss_norm_type == "token":
            return token_nll.sum() / target_mask.sum().clamp_min(1)
        if self.loss_norm_type == "sequence":
            per_sequence = token_nll.sum(dim=-1) / target_mask.sum(dim=-1).clamp_min(1)
            return per_sequence.mean()
        if self.loss_norm_type == "batch":
            return token_nll.sum() / token_nll.shape[0]
        raise ValueError("Invalid loss_norm_type.")

    def _pad_token_id(self, model: transformers.PreTrainedModel | nn.Module) -> int:
        for owner in (
            self.processing_class,
            getattr(model, "generation_config", None),
            getattr(model, "config", None),
        ):
            pad_token_id = getattr(owner, "pad_token_id", None)
            if pad_token_id is not None:
                return int(pad_token_id)
        return 0

    def _vocab_size(self, model: transformers.PreTrainedModel | nn.Module) -> int:
        for candidate in self._model_candidates(model):
            config = getattr(candidate, "config", None)
            vocab_size = self._config_vocab_size(config)
            if vocab_size is not None:
                return vocab_size

            for get_embedding in ("get_output_embeddings", "get_input_embeddings"):
                embedding_fn = getattr(candidate, get_embedding, None)
                if not callable(embedding_fn):
                    continue
                embedding = embedding_fn()
                vocab_size = self._embedding_vocab_size(embedding)
                if vocab_size is not None:
                    return vocab_size

        tokenizer_vocab_size = getattr(self.processing_class, "vocab_size", None)
        if tokenizer_vocab_size is None and self.processing_class is not None:
            try:
                tokenizer_vocab_size = len(self.processing_class)
            except TypeError:
                tokenizer_vocab_size = None
        if tokenizer_vocab_size is not None:
            return int(tokenizer_vocab_size)

        raise ValueError("Could not infer DiffusionGemma text vocab size from model config or embeddings.")

    @classmethod
    def _model_candidates(cls, model: transformers.PreTrainedModel | nn.Module):
        seen: set[int] = set()
        stack = [model]
        while stack:
            candidate = stack.pop(0)
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            yield candidate

            get_base_model = getattr(candidate, "get_base_model", None)
            if callable(get_base_model):
                stack.append(get_base_model())
            for attr in ("base_model", "model"):
                stack.append(getattr(candidate, attr, None))

    @staticmethod
    def _config_vocab_size(config: Any) -> int | None:
        if config is None:
            return None
        for owner in (getattr(config, "text_config", None), config):
            vocab_size = getattr(owner, "vocab_size", None)
            if vocab_size is not None:
                return int(vocab_size)
        return None

    @staticmethod
    def _embedding_vocab_size(embedding: Any) -> int | None:
        if embedding is None:
            return None
        num_embeddings = getattr(embedding, "num_embeddings", None)
        if num_embeddings is not None:
            return int(num_embeddings)
        weight = getattr(embedding, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
            return int(weight.shape[0])
        out_features = getattr(embedding, "out_features", None)
        if out_features is not None:
            return int(out_features)
        return None
