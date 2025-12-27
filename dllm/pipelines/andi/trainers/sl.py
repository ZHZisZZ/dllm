from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from dllm.core.trainers import MDLMTrainer, BD3LMTrainer


class BaseAnDSLTrainer:

    def __init__(
        self,
        ar_loss_scale: bool = 1.0,
        diff_loss_scale: bool = 1.0,
    ):
        self.ar_loss_scale = ar_loss_scale
        self.diff_loss_scale = diff_loss_scale

    def compute_ar_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        # TODO:prepend BOS?

        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [b, l, vocab]

        shift_logits = logits[:, :-1, :].contiguous()  # [b, l-1, v]
        shift_labels = labels[:, 1:].contiguous()  # [b, l-1]

        valid_mask = shift_labels != -100  # [b, l-1]
        token_cnt_per_seq = valid_mask.sum(dim=1, keepdim=True)  # [b, 1]

        # Per-token CE (ignore_index makes ignored positions 0 when reduction='none')
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))  # [(b*(l-1)), v]
        flat_labels = shift_labels.view(-1)  # [b*(l-1)]
        flat_loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction="none",
            ignore_index=-100,
        )  # [b*(l-1)]

        token_loss_2d = flat_loss.view(b, l - 1)  # [b, l-1]
        token_loss = token_loss_2d[valid_mask]  # [n_valid]

        # Normalize like MDLMTrainer
        if self.loss_normalization_type == "batch":
            token_loss_normalized = token_loss / b
        elif self.loss_normalization_type == "sequence":
            denom = token_cnt_per_seq.expand(-1, l - 1)[valid_mask].clamp_min(1)
            token_loss_normalized = token_loss / denom / b
        elif self.loss_normalization_type == "token":
            token_loss_normalized = token_loss / token_cnt_per_seq.sum().clamp_min(1)
        else:
            raise ValueError("Invalid loss_normalization_type.")

        loss = token_loss_normalized.sum()
        return (loss, outputs) if return_outputs else loss

    def compute_diff_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        raise NotImplementedError

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        ar_loss = None
        diff_loss = None
        outputs = None
        log_dict = {}

        # ---------- AR loss ----------
        if self.ar_loss_scale != 0:
            with transformers.modeling_utils.unwrap_model(model).mode_ctx("ar"):
                ar_loss, ar_outputs = self.compute_ar_loss(
                    model, inputs, return_outputs=True, **kwargs
                )
            log_dict["ar_loss"] = ar_loss.item()
            outputs = ar_outputs

        # ---------- Diff loss ----------
        if self.diff_loss_scale != 0:
            with transformers.modeling_utils.unwrap_model(model).mode_ctx("diff"):
                diff_loss, diff_outputs = self.compute_diff_loss(
                    model, inputs, return_outputs=True, **kwargs
                )
            log_dict["diff_loss"] = diff_loss.item()
            outputs = diff_outputs

        # ---------- Combine ----------
        loss = 0.0
        if ar_loss is not None:
            loss = loss + self.ar_loss_scale * ar_loss
        if diff_loss is not None:
            loss = loss + self.diff_loss_scale * diff_loss

        # ---------- logging (only at logging step) ----------
        if (ar_loss is not None) or (diff_loss is not None):
            state = self.state
            args = self.args
            if (
                state.global_step > 0
                and args.logging_steps > 0
                and state.global_step % args.logging_steps == 0
            ):
                if ar_loss is not None:
                    ar_loss_mean = (
                        self.accelerator.gather(ar_loss.detach().float()).mean().item()
                    )
                    log_dict["ar_loss"] = ar_loss_mean

                if diff_loss is not None:
                    diff_loss_mean = (
                        self.accelerator.gather(diff_loss.detach().float())
                        .mean()
                        .item()
                    )
                    log_dict["diff_loss"] = diff_loss_mean

                loss_mean = self.accelerator.gather(loss.detach().float()).mean().item()
                log_dict["loss"] = loss_mean

                self.log(log_dict)

        return (loss, outputs) if return_outputs else loss


class MDLMAnDSLTrainer(BaseAnDSLTrainer, MDLMTrainer):

    def __init__(
        self,
        ar_loss_scale: float = 1.0,
        diff_loss_scale: float = 1.0,
        *args,
        **kwargs,
    ):
        BaseAnDSLTrainer.__init__(
            self, ar_loss_scale=ar_loss_scale, diff_loss_scale=diff_loss_scale
        )
        MDLMTrainer.__init__(self, *args, **kwargs)

    def compute_diff_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        return MDLMTrainer.compute_loss(self, model, inputs, return_outputs, **kwargs)


class BD3LMAnDTrainer(BaseAnDSLTrainer, BD3LMTrainer):

    def __init__(
        self,
        ar_loss_scale: float = 1.0,
        diff_loss_scale: float = 1.0,
        *args,
        **kwargs,
    ):
        BaseAnDSLTrainer.__init__(
            self, ar_loss_scale=ar_loss_scale, diff_loss_scale=diff_loss_scale
        )
        BD3LMTrainer.__init__(self, *args, **kwargs)

    def compute_diff_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        return BD3LMTrainer.compute_loss(self, model, inputs, return_outputs, **kwargs)
