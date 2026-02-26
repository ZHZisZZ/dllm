"""
Generic BD3LM eval base: inherit BaseEvalHarness, override sampler hooks for BD3LMSampler.
generate_until scaffolding is inherited from BaseEvalHarness.
loglikelihood via Monte Carlo ELBO with block-diffusion [x_t ⊕ x_0] attention.
Pipelines (e.g. a2d) import and register with @register_model.

Run: Not runnable directly; use pipeline eval entrypoints (e.g. dllm.pipelines.a2d.eval).
"""

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from tqdm import tqdm

from dllm.core.eval.base import BaseEvalConfig, BaseEvalHarness
from dllm.core.samplers import BD3LMSampler, BD3LMSamplerConfig
from dllm.core.trainers.bd3lm import _create_bd3lm_attention_mask


@dataclass
class BD3LMEvalSamplerConfig(BD3LMSamplerConfig):
    """Default sampler config for BD3LM eval."""

    max_new_tokens: int = 128
    steps: int = 128
    block_size: int = 32


@dataclass
class BD3LMEvalConfig(BaseEvalConfig):
    """Eval-only config for BD3LM."""

    max_length: int = 2048
    batch_size: int = 32
    mc_num: int = 128
    is_check_greedy: bool = False


class BD3LMEvalHarness(BaseEvalHarness):
    """
    BD3LM eval: loglikelihood (Monte Carlo ELBO) + generate_until (inherited).
    Constructs [x_t ⊕ x_0] with block-diffusion attention (M_BD | M_OBC | M_BC).
    """

    def __init__(
        self,
        eval_config: BD3LMEvalConfig | None = None,
        sampler_config: BD3LMSamplerConfig | None = None,
        sampler_cls: type[BD3LMSampler] = BD3LMSampler,
        **kwargs,
    ):
        eval_config = eval_config or BD3LMEvalConfig()
        sampler_config = sampler_config or BD3LMEvalSamplerConfig()

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )

        self.mask_id = self.tokenizer.mask_token_id
        self.max_length = int(kwargs.get("max_length", eval_config.max_length))
        self.mc_num = int(kwargs.get("mc_num", eval_config.mc_num))
        self.is_check_greedy = kwargs.get(
            "is_check_greedy", eval_config.is_check_greedy
        )
        self.block_size = int(kwargs.get("block_size", sampler_config.block_size))

        assert self.mc_num % self.batch_size == 0

    # ── Private helpers (low-level → high-level) ───────────────────────

    def _encode_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        """Encode context and continuation; move trailing spaces from context to continuation."""
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def _create_attention_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Create the BD3LM block-diffusion attention mask for [x_t ⊕ x_0]."""
        unwrapped = (
            self.accelerator.unwrap_model(self.model)
            if self.accelerator
            else self.model
        )
        attn_impl = unwrapped.config._attn_implementation

        if attn_impl == "sdpa":
            mask = _create_bd3lm_attention_mask(
                b=None,
                h=None,
                q_idx=torch.arange(seq_len * 2)[:, None],
                kv_idx=torch.arange(seq_len * 2)[None, :],
                block_size=self.block_size,
                n=seq_len,
            )
            mask = mask.unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * seq_len, 2 * seq_len)
            return mask.to(device)
        elif attn_impl == "flex_attention":
            from torch.nn.attention.flex_attention import create_block_mask

            return create_block_mask(
                partial(
                    _create_bd3lm_attention_mask,
                    block_size=self.block_size,
                    n=seq_len,
                ),
                B=None,
                H=None,
                Q_LEN=seq_len * 2,
                KV_LEN=seq_len * 2,
            )
        else:
            raise NotImplementedError(
                f"Unsupported attention implementation: {attn_impl}"
            )

    @torch.no_grad()
    def _get_logits(
        self, batch: torch.Tensor, x0: torch.Tensor, prompt_index: torch.Tensor
    ) -> torch.Tensor:
        """BD3LM forward: [x_t ⊕ x_0] with block-diffusion attention, return x_t logits."""
        b, l = batch.shape

        # [x_t ⊕ x_0]: noised first half, clean second half
        concat_input_ids = torch.cat([batch, x0], dim=1)  # [b, 2l]

        # Position IDs: [0..l-1, 0..l-1] (duplicated for both halves)
        base_pos = torch.arange(l, device=batch.device).unsqueeze(0).expand(b, l)
        concat_position_ids = torch.cat([base_pos, base_pos], dim=1)  # [b, 2l]

        # Block-diffusion attention mask
        attention_mask = self._create_attention_mask(l, batch.device)

        logits = self.model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=concat_position_ids,
        ).logits

        return logits[:, :l]  # Only x_t half predictions

    def _forward_process(
        self, batch: torch.Tensor, prompt_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply forward diffusion process by masking a random subset of target tokens."""
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(
            torch.linspace(
                float(k),
                k + (b - 1) * (target_len / b),
                steps=b,
                device=batch.device,
            )
        ).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat(
            (
                torch.zeros(
                    b,
                    int(prompt_index.sum()),
                    dtype=torch.bool,
                    device=batch.device,
                ),
                is_mask,
            ),
            dim=1,
        )

        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        p_mask = (x / target_len).unsqueeze(1).repeat(1, l)
        return noisy_batch, p_mask

    @torch.no_grad()
    def _get_loglikelihood(self, prefix: torch.Tensor, target: torch.Tensor) -> float:
        """Monte Carlo estimate of log-likelihood via _forward_process + _get_logits."""
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        # Clean x_0 for [x_t ⊕ x_0] construction
        x0 = seq.clone()

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
            mask_indices = perturbed_seq == self.mask_id
            logits = self._get_logits(perturbed_seq, x0, prompt_index)
            loss = (
                F.cross_entropy(
                    logits[mask_indices],
                    seq[mask_indices],
                    reduction="none",
                )
                / p_mask[mask_indices]
            )
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _suffix_greedy_prediction(
        self, prefix: torch.Tensor, target: torch.Tensor
    ) -> bool:
        """Greedy unmasking check via _get_logits."""
        if not self.is_check_greedy:
            return False

        seq = torch.full(
            (1, len(prefix) + len(target)),
            self.mask_id,
            device=self.device,
        )
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, : len(prefix)] = prefix

        # Clean reference for [x_t ⊕ x_0] construction
        x0_clean = torch.cat([prefix, target]).unsqueeze(0)

        for i in range(len(target)):
            mask_index = seq == self.mask_id
            logits = self._get_logits(seq, x0_clean, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)
            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(
                dim=-1
            )
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix) :]
        return torch.all(correct).item()

    # ── Public API (lm-eval interface) ────────────────────────────────

    @torch.no_grad()
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        out = []
        for instance in tqdm(requests, desc="Computing likelihood..."):
            context_enc, continuation_enc = self._encode_pair(*instance.args)
            assert len(context_enc) + len(continuation_enc) <= self.max_length, (
                f"Context + continuation length exceeds "
                f"{self.max_length} tokens: "
                f"{len(context_enc)} + {len(continuation_enc)}"
            )

            context = torch.tensor(context_enc, device=self.device, dtype=torch.long)
            continuation = torch.tensor(
                continuation_enc, device=self.device, dtype=torch.long
            )

            if continuation.shape[0] == 0:
                out.append((0.0, False))
                continue

            logprob = self._get_loglikelihood(context, continuation)
            isgreedy = self._suffix_greedy_prediction(context, continuation)
            out.append((logprob, isgreedy))
        torch.cuda.empty_cache()
        return out
