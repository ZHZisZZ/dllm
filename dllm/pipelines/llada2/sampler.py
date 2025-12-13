"""
Block diffusion-style sampler for LLaDA2-MoE.

This mirrors the blockwise masked-denoising generate logic from
`dllm.pipelines.llada2.models.modeling_llada2_moe`, but as a standalone sampler that follows the
BaseSampler interface (similar to bd3lm/mdlm).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, SamplerConfig, SamplerOutput


def even_transfer_schedule(block_size: int, steps_per_block: int) -> torch.Tensor:
    """
    Evenly split `block_size` tokens across `steps_per_block` steps.
    Example: block_size=32, steps=32 -> 32 ones.
    """
    if steps_per_block <= 0:
        return torch.tensor([], dtype=torch.int64)
    base = block_size // steps_per_block
    remainder = block_size % steps_per_block
    schedule = torch.full((steps_per_block,), base, dtype=torch.int64)
    schedule[:remainder] += 1
    return schedule


def top_k_top_p(
    logits: torch.Tensor, top_k: Optional[int], top_p: Optional[float]
) -> torch.Tensor:
    """Filter logits with top-k / top-p; returns filtered logits."""
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_values = values[..., -1, None]
        logits = torch.where(
            logits < min_values, torch.full_like(logits, float("-inf")), logits
        )

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask = torch.full_like(logits, False, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))

    return logits


def sample_tokens(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample one token per position; returns sampled ids and their probabilities.
    """
    if temperature is None or temperature == 0.0:
        filtered = top_k_top_p(logits, top_k, top_p)
        probs = F.softmax(filtered, dim=-1)
        tokens = torch.argmax(probs, dim=-1)
        token_prob = torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
        return tokens, token_prob

    logits = logits / temperature
    filtered = top_k_top_p(logits, top_k, top_p)
    probs = F.softmax(filtered, dim=-1)
    tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(
        *probs.shape[:-1]
    )
    token_prob = torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
    return tokens, token_prob


@dataclass
class LLaDA2SamplerConfig(SamplerConfig):
    max_new_tokens: int = 128
    block_size: int = 32
    steps_per_block: int = 32
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    threshold: float = 0.95
    minimal_topk: int = 1
    eos_early_stop: bool = False


@dataclass
class LLaDA2Sampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: LLaDA2SamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        """
        Block diffusion-like sampler that mirrors `LLaDA2MoeModelLM.generate`.
        Currently supports equal-length prompts.
        """
        if config is None:
            config = LLaDA2SamplerConfig()

        block_size = kwargs.get("block_size", config.block_size)
        steps_per_block = kwargs.get("steps_per_block", config.steps_per_block)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        temperature = kwargs.get("temperature", config.temperature)
        top_p = kwargs.get("top_p", config.top_p)
        top_k = kwargs.get("top_k", config.top_k)
        threshold = kwargs.get("threshold", config.threshold)
        minimal_topk = kwargs.get("minimal_topk", config.minimal_topk)
        eos_early_stop = kwargs.get("eos_early_stop", config.eos_early_stop)
        return_dict = kwargs.get("return_dict", config.return_dict)

        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        # Normalize inputs
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]
        if len(set(prompt_lens)) != 1:
            raise ValueError(
                "LLaDA2Sampler expects all prompts to have the same length."
            )

        prompt_len = prompt_lens[0]
        steps_per_block = min(
            steps_per_block,
            max_new_tokens // minimal_topk if minimal_topk > 0 else steps_per_block,
        )

        num_blocks = (prompt_len + max_new_tokens + block_size - 1) // block_size
        total_len = num_blocks * block_size

        # Block-wise attention mask (block causal, bidirectional within block)
        block_mask = torch.tril(
            torch.ones(num_blocks, num_blocks, device=self.model.device)
        )
        block_attn = (
            (
                block_mask.repeat_interleave(block_size, dim=0)
                .repeat_interleave(block_size, dim=1)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            .log()
            .to(torch.bfloat16)
        )

        position_ids = torch.arange(total_len, device=self.model.device).unsqueeze(0)

        # Canvas initialized with masks, prompts filled at the front
        x = torch.full(
            (len(inputs), total_len),
            mask_id,
            dtype=torch.long,
            device=self.model.device,
        )
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p

        prompt_blocks = prompt_len // block_size
        denoising_steps_per_block = steps_per_block
        transfer_schedule = even_transfer_schedule(
            block_size, denoising_steps_per_block
        )

        histories = [x.clone()] if return_dict else None

        for blk in range(prompt_blocks, num_blocks):
            window_end = (blk + 1) * block_size
            cur_attn = block_attn[:, :, :window_end, :window_end]
            cur_pos = position_ids[:, :window_end]

            for step_idx in range(denoising_steps_per_block):
                block_slice = x[:, window_end - block_size : window_end]
                active_mask = block_slice == mask_id
                if not active_mask.any():
                    break

                logits = self.model(
                    x[:, :window_end],
                    attention_mask=cur_attn,
                    position_ids=cur_pos,
                ).logits

                logits_block = logits[:, -block_size:, :]
                tokens, probs = sample_tokens(
                    logits_block, temperature=temperature, top_k=top_k, top_p=top_p
                )

                num_to_transfer = int(transfer_schedule[step_idx].item())
                transfer_index = torch.zeros_like(block_slice, dtype=torch.bool)

                for b in range(block_slice.size(0)):
                    conf = torch.where(
                        active_mask[b],
                        probs[b],
                        torch.full_like(probs[b], -float("inf")),
                    )
                    high_conf = (conf > threshold) & active_mask[b]
                    if high_conf.sum().item() >= num_to_transfer:
                        transfer_index[b] = high_conf
                    else:
                        if num_to_transfer > 0 and active_mask[b].any():
                            k = min(num_to_transfer, active_mask[b].sum().item())
                            _, idx = torch.topk(conf, k=k)
                            transfer_index[b, idx] = True

                block_slice[transfer_index] = tokens[transfer_index]

                if histories is not None:
                    histories.append(x.clone())

                if eos_early_stop and eos_id is not None:
                    if (block_slice == eos_id).any():
                        break

        if not return_dict:
            return x
        return SamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor, list],
        config: LLaDA2SamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput:
        raise NotImplementedError
