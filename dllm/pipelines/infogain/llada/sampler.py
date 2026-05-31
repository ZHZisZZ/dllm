"""
Info-Gain LLaDA sampler (no KV cache path), with Fast-dLLM KV cache decoding when ``use_cache`` is set.

Reference: https://github.com/yks23/Information-Gain-Sampler
Fast-dLLM cache path: https://github.com/NVlabs/Fast-dLLM

Run: python -u examples/infogain/llada/sample.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSamplerOutput
from dllm.pipelines.fastdllm.llada.sampler import (
    FastdLLMLLaDASampler,
    FastdLLMLLaDASamplerConfig,
)
from dllm.pipelines.infogain import info_gain_ops as ig


@dataclass
class InfoGainLLaDASamplerConfig(FastdLLMLLaDASamplerConfig):
    """Info-Gain fields (used when ``use_cache`` is ``None``)."""

    candidate_number: int = 8
    position_temperature: float = 0.2
    variant: str = "info_gain"  # "info_gain" | "lookum"


class InfoGainLLaDASampler(FastdLLMLLaDASampler):
    """LLaDA sampler: Info-Gain when ``use_cache=None``; otherwise Fast-dLLM KV-cache decoding."""

    @torch.no_grad()
    def sample(
        self,
        inputs: Union[List[torch.Tensor], List[List[int]], torch.Tensor],
        config: Optional[InfoGainLLaDASamplerConfig] = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        cfg = config or InfoGainLLaDASamplerConfig()
        use_cache = kwargs.get("use_cache", cfg.use_cache)
        if use_cache == "none":
            use_cache = None
        if use_cache is not None:
            return super().sample(inputs, config=cfg, **kwargs)
        return self._sample_info_gain_no_cache(inputs, cfg, **kwargs)

    @torch.no_grad()
    def _sample_info_gain_no_cache(
        self,
        inputs: Union[List[torch.Tensor], List[List[int]], torch.Tensor],
        config: InfoGainLLaDASamplerConfig,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )
        threshold = kwargs.get("threshold", config.threshold)
        candidate_number = kwargs.get("candidate_number", config.candidate_number)
        position_temperature = kwargs.get(
            "position_temperature", config.position_temperature
        )
        variant = kwargs.get("variant", config.variant)

        steps = int(steps)
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        candidate_number = int(candidate_number)
        position_temperature = float(position_temperature)
        if candidate_number < 1:
            raise ValueError(
                f"candidate_number must be >= 1, got {candidate_number}"
            )
        if position_temperature < 0:
            raise ValueError(
                "position_temperature must be non-negative, "
                f"got {position_temperature}"
            )
        ig.validate_info_gain_variant(variant)

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            inputs_list = [
                row.to(device=self.model.device, dtype=torch.long) for row in inputs
            ]
        else:
            if len(inputs) == 0:
                raise ValueError("inputs is empty")
            if isinstance(inputs[0], list):
                inputs_list = [
                    torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                    for p in inputs  # type: ignore[arg-type]
                ]
            else:
                inputs_list = [
                    p.to(device=self.model.device, dtype=torch.long) for p in inputs  # type: ignore[arg-type]
                ]

        prompt_lens = [p.shape[0] for p in inputs_list]
        B = len(inputs_list)
        max_prompt_len = max(prompt_lens)

        if right_shift_logits:
            fixed = []
            for p in inputs_list:
                if p.numel() == 0:
                    fixed.append(
                        torch.tensor(
                            [bos_id], device=self.model.device, dtype=torch.long
                        )
                    )
                else:
                    fixed.append(p)
            inputs_list = fixed
            prompt_lens = [p.shape[0] for p in inputs_list]
            max_prompt_len = max(prompt_lens)

        max_new_tokens, max_length = ig.resolve_generation_lengths(
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            max_prompt_len=max_prompt_len,
        )
        block_size = max_new_tokens if block_size is None else int(block_size)
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")

        if B != 1:
            raise ValueError(
                "Info-Gain LLaDA decoding (use_cache=None) currently supports batch size 1. "
                "Set eval batch_size=1, or use the Fast-dLLM path with use_cache='prefix' "
                "or 'dual' when batching equal-length prompts."
            )

        T = int(max_length)
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)

        for i, p in enumerate(inputs_list):
            pl = p.shape[0]
            x[i, :pl] = p
            gen_end = min(pl + max_new_tokens, T)
            x[i, pl:gen_end] = mask_id
            attention_mask[i, :gen_end] = 1

        histories = [x.clone()] if return_dict else None

        num_blocks = max(1, math.ceil(max_new_tokens / block_size))
        steps_per_block = max(1, math.ceil(steps / num_blocks))

        def _apply_suppressions(logits_: torch.Tensor):
            if suppress_tokens:
                for tid in suppress_tokens:
                    logits_[:, :, tid] = -torch.inf
            if begin_suppress_tokens:
                for tid in begin_suppress_tokens:
                    logits_[:, :, tid] = -torch.inf

        for b in range(num_blocks):
            widths: list[tuple[int, int, int]] = []
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )
            for j in range(B):
                start_j = prompt_lens[j] + b * block_size
                end_j = min(start_j + block_size, prompt_lens[j] + max_new_tokens, T)
                width_j = max(0, end_j - start_j)
                widths.append((start_j, end_j, width_j))
                if width_j > 0:
                    block_mask_index[j, :width_j] = x[j, start_j:end_j] == mask_id

            for _step in range(steps_per_block):
                mask_allowed = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    start_j, end_j, width_j = widths[j]
                    if width_j > 0:
                        mask_allowed[j, start_j:end_j] = x[j, start_j:end_j] == mask_id

                if mask_allowed.sum() == 0:
                    break

                out = self.model(x, attention_mask=attention_mask)
                logits = out.logits
                _apply_suppressions(logits)
                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                row = 0
                start_j, end_j, width_j = widths[row]
                if width_j == 0:
                    continue
                bs, be = start_j, end_j
                lx = logits[row : row + 1]
                xx = x[row : row + 1]
                ma = mask_allowed[row : row + 1]

                probs = F.softmax(lx[:, bs:be].float(), dim=-1)
                max_conf = probs.max(-1).values
                block_masked = ma[0, bs:be]
                if (
                    threshold is not None
                    and threshold > 0
                    and block_masked.any()
                    and (max_conf[0][block_masked] >= threshold).all()
                ):
                    xx_new = ig.greedy_unmask_block(
                        xx, lx, ma, bs, be, temperature, mask_id
                    )
                    x[row : row + 1] = xx_new
                    continue

                k = 1
                result = ig.generate_candidates(
                    lx,
                    xx,
                    ma,
                    bs,
                    be,
                    k,
                    candidate_number,
                    temperature,
                    position_temperature,
                )
                actions, x0s, conf_base, valid, _ = result

                if actions is None:
                    if valid.shape[0] == 0:
                        continue
                    best_pos = valid[conf_base[0, valid].argmax()].unsqueeze(0)
                    x_row = x[row].clone()
                    x_row[best_pos] = x0s[0, best_pos]
                    x[row] = x_row
                    continue

                nc = len(actions)
                x_batch = xx.expand(nc, -1).clone()
                for i, (act, x0) in enumerate(zip(actions, x0s)):
                    x_batch[i, act] = x0[0, act]

                attn = attention_mask[row : row + 1].expand(nc, -1)
                next_logits = self.model(x_batch, attention_mask=attn).logits
                _apply_suppressions(next_logits)
                if right_shift_logits:
                    next_logits = torch.cat(
                        [next_logits[:, :1], next_logits[:, :-1]], dim=1
                    )

                _, _, J = ig.score_candidates(
                    lx, next_logits, x_batch, actions, mask_id, variant
                )
                best = int(J.argmax().item())
                x_row = x[row].clone()
                act = actions[best]
                x_row[act] = x0s[best][0, act]
                x[row] = x_row

                if histories is not None:
                    histories.append(x.clone())

        if not return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self,
        inputs: Union[List[torch.Tensor], List[List[int]]],
        config: InfoGainLLaDASamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput:
        raise NotImplementedError
