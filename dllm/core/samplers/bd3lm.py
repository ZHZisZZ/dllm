"""
reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, SamplerConfig, SamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


def _prepare_for_sampling(
    x: torch.Tensor,
    block_size: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a block-wise bidirectional attention mask and position_ids
    over the entire sequence (prompt + generated).

    Padding tokens (pad_token_id) are excluded from attention: they are neither
    valid queries nor valid keys.

    Block boundaries are defined in *physical* coordinates (shared across batch):
      - block_id[pos] = pos // block_size   for column index pos = 0..T-1

    For position_ids (used by RoPE), we still use per-sample logical positions:
      - valid[b, t] = (x[b, t] != pad_token_id)
      - pos_raw[b, t]  = count of valid tokens up to and including t (1-based)
      - logical_pos[b, t] = pos_raw[b, t] - 1, for valid positions

    Returns:
        attn_mask: [B, 1, T, T] bool
        position_ids: [B, T] long, logical positions (padding set to 0)
    """
    B, T = x.shape
    device = x.device

    # Per-sample valid mask
    valid = x != pad_token_id  # [B, T]

    # Per-sample logical positions for RoPE (skip padding)
    pos_raw = torch.cumsum(valid.to(torch.long), dim=-1)  # [B, T], 1-based
    logical_pos = pos_raw - 1  # [B, T], 0-based

    # Position ids: logical positions for valid tokens, 0 for padding
    position_ids = torch.where(
        valid,
        logical_pos,
        torch.zeros_like(logical_pos),
    ).to(
        device=device, dtype=torch.long
    )  # [B, T]

    # Block ids for attention: defined in physical coordinates
    pos = torch.arange(T, device=device)  # [T]
    block_ids = torch.div(pos, block_size, rounding_mode="floor")  # [T]
    block_ids = block_ids.view(1, T).expand(B, -1)  # [B, T]

    # Mark padding positions as "no block"
    block_ids = torch.where(
        valid,
        block_ids,
        torch.full_like(block_ids, -1),
    )

    # Build [B, 1, T, T] mask
    bid_q = block_ids.view(B, 1, T, 1)  # query
    bid_k = block_ids.view(B, 1, 1, T)  # key

    valid_q = bid_q >= 0
    valid_k = bid_k >= 0

    base_mask = bid_k <= bid_q
    attn_mask = base_mask & valid_q & valid_k  # [B, 1, T, T]

    return attn_mask, position_ids


def _diffusion_step_block(
    logits: torch.Tensor,  # [B, L, V]
    x_block: torch.Tensor,  # [B, L]
    mask_block: torch.Tensor,  # [B, L] bool
    num_transfer_step: torch.Tensor,  # [B]
    temperature: float,
    remasking: str,
) -> torch.Tensor:
    """
    One diffusion step over a block slice [B, L].
    """
    B, L, _ = logits.shape
    device = logits.device

    if not mask_block.any():
        return x_block

    # Gumbel-max sampling
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]

    # Confidence
    if remasking == "low_confidence":
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # [B, L]
    elif remasking == "random":
        x0_p = torch.rand((B, L), device=device)
    else:
        raise NotImplementedError(remasking)

    # Only masked positions can change
    x0 = torch.where(mask_block, x0, x_block)
    neg_inf = torch.full_like(x0_p, -float("inf"))
    confidence = torch.where(mask_block, x0_p, neg_inf)

    # Pick positions to commit
    transfer = torch.zeros_like(x0, dtype=torch.bool)  # [B, L]
    for j in range(B):
        k = int(num_transfer_step[j].item())
        if k <= 0:
            continue
        valid_count = (confidence[j] > -float("inf")).sum().item()
        if valid_count == 0:
            continue
        k = min(k, valid_count)
        _, sel = torch.topk(confidence[j], k)
        transfer[j, sel] = True

    x_block_new = x_block.clone()
    x_block_new[transfer] = x0[transfer]
    return x_block_new


@dataclass
class BD3LMSamplerConfig(SamplerConfig):
    max_new_tokens: int = 128
    max_length: int = (
        None  # There's no explicit length_limit except for the tokenizer/model context
    )
    block_size: int = 32
    steps: int = 128
    steps_per_block: int | None = None
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    right_shift_logits: bool = False


@dataclass
class BD3LMSampler(BaseSampler):

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: BD3LMSamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        """
        Generate text using block diffusion language modeling.

        Generates text block-by-block with an attention pattern, where each
        block undergoes multiple diffusion steps before moving to the next block.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            SamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """

        if config is None:
            config = BD3LMSamplerConfig()

        # ---- pull args from config, allow kwargs to override ----
        steps = kwargs.get("steps", config.steps)
        steps_per_block = kwargs.get("steps_per_block", config.steps_per_block)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)

        assert block_size >= 1
        assert steps >= 1

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        pad_id = self.tokenizer.pad_token_id  # used as padding here
        eos_id = self.tokenizer.eos_token_id

        # ---- normalize inputs to tensors ----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        prompt_lens = [p.shape[0] for p in inputs]

        # Decide how many new tokens to generate
        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        max_prompt_len = max(prompt_lens)

        # ==========================================================
        # 1) Initialize with prompt only (left padded with pad_id)
        #    pad prefix length to a multiple of block_size
        # ==========================================================
        padded_prompt_len = (
            (max_prompt_len + block_size - 1) // block_size
        ) * block_size

        x = torch.full(
            (B, padded_prompt_len),
            pad_id,
            dtype=torch.long,
            device=self.model.device,
        )
        for b, p in enumerate(inputs):
            L = prompt_lens[b]
            offset = padded_prompt_len - L  # left padding
            x[b, offset : offset + L] = p

        # Tokens considered "given" for unconditional branch in CFG.
        unmasked_index = (x != mask_id) & (x != pad_id)
        if cfg_keep_tokens:
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & (~keep_mask)

        # track done per sequence (EOS)
        done = torch.zeros((B,), dtype=torch.bool, device=self.model.device)

        # ---- block scheduling ----
        num_blocks = math.ceil(max_new_tokens / block_size)
        if steps_per_block is None:
            steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None

        generated = 0  # number of generated tokens so far

        # ==========================================================
        # 2) Block-by-block generation loop
        # ==========================================================
        for b_idx in range(num_blocks):
            if done.all():
                break

            T_prefix = x.shape[1]  # current total length before appending this block

            # With padded_prompt_len aligned, we always append whole blocks (except possibly final)
            cur_block_len = min(block_size, max_new_tokens - generated)
            if cur_block_len <= 0:
                break

            # ------------------------------------------------------
            # 2.1) Prefix: prompt + all previous blocks
            # ------------------------------------------------------
            x_prefix = x  # [B, T_prefix]
            B_cur, T_prefix = x_prefix.shape

            prefix_attn, prefix_pos = _prepare_for_sampling(
                x=x_prefix,
                block_size=block_size,
                pad_token_id=pad_id,
            )  # [B,1,T_prefix,T_prefix], [B,T_prefix]

            # Conditional prefix cache + last logits
            out_prefix = self.model(
                x_prefix,
                attention_mask=prefix_attn,
                position_ids=prefix_pos,
                use_cache=True,
            )
            cond_past = out_prefix.past_key_values
            cond_prefix_last_logits = out_prefix.logits[:, -1:, :]  # [B, 1, V]

            # Unconditional prefix cache + last logits (if CFG enabled)
            if cfg_scale > 0.0:
                un_x_prefix = x_prefix.clone()
                un_x_prefix[unmasked_index] = mask_id

                out_un_prefix = self.model(
                    un_x_prefix,
                    attention_mask=prefix_attn,
                    position_ids=prefix_pos,
                    use_cache=True,
                )
                uncond_past = out_un_prefix.past_key_values
                uncond_prefix_last_logits = out_un_prefix.logits[:, -1:, :]  # [B, 1, V]
            else:
                uncond_past = None
                uncond_prefix_last_logits = None

            # ------------------------------------------------------
            # 2.2) Append new block of mask tokens to the right
            # ------------------------------------------------------
            new_block = torch.full(
                (B, cur_block_len), mask_id, dtype=torch.long, device=self.model.device
            )
            # if done.any():
            #     new_block = torch.where(
            #         done.unsqueeze(1),
            #         torch.full_like(new_block, pad_id),
            #         new_block,
            #     )

            x = torch.cat([x, new_block], dim=1)  # [B, T_prefix + cur_block_len]

            unmasked_index = torch.cat(
                [
                    unmasked_index,
                    torch.zeros(
                        (B, cur_block_len), dtype=torch.bool, device=self.model.device
                    ),
                ],
                dim=1,
            )

            B_cur, T_total = x.shape

            block_mask_index = x[:, -cur_block_len:] == mask_id  # [B, cur_block_len]

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            # Full attention mask + pos for prefix + current block
            full_attention_mask, full_position_ids = _prepare_for_sampling(
                x=x,
                block_size=block_size,
                pad_token_id=pad_id,
            )  # [B,1,T_total,T_total], [B,T_total]

            # Block view
            attn_block = full_attention_mask[
                :, :, T_prefix:T_total, :
            ]  # [B,1,L_q,T_total]
            pos_block = full_position_ids[:, T_prefix:T_total]  # [B,L_q]

            # ======================================================
            # 3) Inner diffusion loop within the current block
            # ======================================================
            for i_step in range(effective_steps):
                x_block = x[:, T_prefix:T_total]  # [B, cur_block_len]
                mask_block = x_block == mask_id

                if not mask_block.any():
                    break

                # ---- Conditional logits for current block ----
                cond_logits_block = self.model(
                    x_block,
                    attention_mask=attn_block,
                    position_ids=pos_block,
                    past_key_values=copy.deepcopy(cond_past),
                    use_cache=False,
                ).logits  # [B, cur_block_len, V]

                logits_block = cond_logits_block

                # ---- Optional CFG ----
                if cfg_scale > 0.0:
                    un_logits_block = self.model(
                        x_block,
                        attention_mask=attn_block,
                        position_ids=pos_block,
                        past_key_values=copy.deepcopy(uncond_past),
                        use_cache=False,
                    ).logits  # [B, cur_block_len, V]

                    logits_block = un_logits_block + (cfg_scale + 1.0) * (
                        cond_logits_block - un_logits_block
                    )

                # ---- Global AR-style right shift across blocks ----
                if right_shift_logits:
                    if cfg_scale > 0.0:
                        prefix_last_logits = uncond_prefix_last_logits + (
                            cfg_scale + 1.0
                        ) * (
                            cond_prefix_last_logits - uncond_prefix_last_logits
                        )  # [B, 1, V]
                    else:
                        prefix_last_logits = cond_prefix_last_logits  # [B, 1, V]

                    shifted = torch.empty_like(logits_block)
                    shifted[:, 0:1, :] = prefix_last_logits
                    shifted[:, 1:, :] = logits_block[:, :-1, :]
                    logits_block = shifted

                # ---- One diffusion step over this block ----
                x_block_updated = _diffusion_step_block(
                    logits=logits_block,
                    x_block=x_block,
                    mask_block=mask_block,
                    num_transfer_step=num_transfer_tokens[:, i_step],
                    temperature=temperature,
                    remasking=remasking,
                )

                # Write back
                x[:, T_prefix:T_total] = x_block_updated

                if histories is not None:
                    histories.append(x.clone())

            # per-sequence EOS stopping (after finishing denoising the block)
            if eos_id is not None:
                eos_in_block = (x[:, T_prefix:T_total] == eos_id).any(dim=1)
                done = done | eos_in_block

            generated += cur_block_len

        # ==========================================================
        # 4) Output
        # ==========================================================
        if not return_dict:
            return x
        else:
            return SamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: SamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput:
        raise NotImplementedError
