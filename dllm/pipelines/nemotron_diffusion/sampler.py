"""
Nemotron-Labs diffusion sampler for the dllm sampler interface.

This keeps the official HF remote-code generation order, but lays it out like
the DiffusionGemma sampler: one visible `sample()` loop plus small math helpers.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise


def sample_tokens(
    logits, temperature, mask_index, x, num_transfer_tokens, threshold=None
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -torch.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


@dataclass
class NemotronDiffusionSamplerOutput(BaseSamplerOutput):
    nfe: int | None = None


@dataclass
class NemotronDiffusionSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 8192
    block_length: int = 8
    threshold: float | None = 0.9
    temperature: float = 0.0
    eos_token_id: int | None = None
    max_thinking_tokens: int | None = 6000


@dataclass
class NemotronDiffusionSampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list] | torch.Tensor,
        config: NemotronDiffusionSamplerConfig | None = None,
        **kwargs,
    ) -> NemotronDiffusionSamplerOutput | torch.Tensor:
        if config is None:
            config = NemotronDiffusionSamplerConfig()

        max_new_tokens = int(kwargs.get("max_new_tokens", config.max_new_tokens))
        block_length = int(kwargs.get("block_length", config.block_length))
        threshold = kwargs.get("threshold", config.threshold)
        temperature = float(kwargs.get("temperature", config.temperature))
        eos_token_id = kwargs.get("eos_token_id", config.eos_token_id)
        max_thinking_tokens = kwargs.get(
            "max_thinking_tokens", config.max_thinking_tokens
        )
        return_dict = bool(kwargs.get("return_dict", config.return_dict))

        model = self.model
        device = torch.device(model.device)
        prompt_ids = (
            inputs.to(device=device, dtype=torch.long)
            if isinstance(inputs, torch.Tensor)
            else torch.as_tensor(inputs, dtype=torch.long, device=device)
        )
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids[None]
        if prompt_ids.dim() != 2 or prompt_ids.numel() == 0:
            raise ValueError(
                "NemotronDiffusionSampler expects a non-empty 1D or 2D prompt tensor."
            )

        assert max_new_tokens % block_length == 0

        if eos_token_id is None:
            eos_token_id = getattr(getattr(model, "config", None), "eos_token_id", None)

        end_think_token_id = None
        if max_thinking_tokens is not None:
            end_think_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
            if end_think_token_id is None or end_think_token_id == getattr(
                self.tokenizer, "unk_token_id", None
            ):
                raise ValueError(
                    "NemotronDiffusionSampler requires tokenizer token `</think>`."
                )

        mask_id = model.mask_token_id
        x_accum = prompt_ids.clone()
        batch_size = prompt_ids.shape[0]
        num_blocks = max_new_tokens // block_length
        nfe = 0

        def set_diffusion_lm(value: bool):
            for layer in model.encoder.layers:
                if hasattr(layer.self_attn, "diffusion_lm"):
                    layer.self_attn.diffusion_lm = value

        set_diffusion_lm(False)
        output = model(prompt_ids, use_cache=True, use_causal_mask=True)
        past_key_values = output.past_key_values
        set_diffusion_lm(True)

        last_logit = output.logits[:, -1, :]
        if temperature > 0:
            next_token = torch.multinomial(
                torch.softmax(last_logit / temperature, dim=-1), num_samples=1
            )
        else:
            next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

        for num_block in range(num_blocks):
            mask_block = torch.full(
                (batch_size, block_length),
                mask_id,
                dtype=prompt_ids.dtype,
                device=prompt_ids.device,
            )
            mask_block[:, 0] = next_token[:, 0]

            x_accum = torch.cat([x_accum, mask_block], dim=1)
            block_start = prompt_ids.size(1) + num_block * block_length
            block_slice = slice(block_start, block_start + block_length)

            if end_think_token_id is not None and max_thinking_tokens is not None:
                tokens_before = num_block * block_length
                tokens_after = tokens_before + block_length
                if tokens_after > max_thinking_tokens:
                    gen_so_far = x_accum[:, prompt_ids.size(1) : block_start]
                    has_end_think = (
                        (gen_so_far == end_think_token_id).any(dim=1)
                        if gen_so_far.size(1) > 0
                        else torch.zeros(
                            batch_size, dtype=torch.bool, device=prompt_ids.device
                        )
                    )
                    if not has_end_think.all():
                        offset = max(0, max_thinking_tokens - tokens_before)
                        inject_pos = block_start + offset
                        for b in range(batch_size):
                            if not has_end_think[b]:
                                x_accum[b, inject_pos] = end_think_token_id

            num_transfer_tokens = _get_num_transfer_tokens(
                x_accum[:, block_slice] == mask_id, block_length
            )

            for i in range(block_length):
                mask_block_idx = x_accum[:, block_slice] == mask_id
                if mask_block_idx.sum() == 0:
                    break

                nfe += 1
                output = model(
                    x_accum[:, block_slice],
                    past_key_values=past_key_values,
                    use_cache=False,
                )

                x0, transfer_idx = sample_tokens(
                    output.logits,
                    temperature,
                    mask_block_idx,
                    x_accum[:, block_slice],
                    num_transfer_tokens=num_transfer_tokens[:, i],
                    threshold=threshold,
                )
                cur = x_accum[:, block_slice].clone()
                cur[transfer_idx] = x0[transfer_idx]
                x_accum[:, block_slice] = cur

                if eos_token_id is not None:
                    block_tokens = x_accum[:, block_slice]
                    eos_mask = block_tokens == eos_token_id
                    if eos_mask.any(dim=1).any():
                        after_eos = eos_mask.cumsum(dim=1).bool()
                        mask_before = (block_tokens == mask_id) & ~after_eos
                        if (eos_mask.any(dim=1) & ~mask_before.any(dim=1)).any():
                            break

            set_diffusion_lm(False)
            output = model(
                x_accum[:, block_slice],
                past_key_values=past_key_values,
                use_cache=True,
                use_causal_mask=True,
            )
            past_key_values = output.past_key_values
            nfe += 1

            set_diffusion_lm(True)
            last_logit = output.logits[:, -1, :]
            if temperature > 0:
                next_token = torch.multinomial(
                    torch.softmax(last_logit / temperature, dim=-1), num_samples=1
                )
            else:
                next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

            if eos_token_id is not None:
                gen_so_far = x_accum[:, prompt_ids.size(1) :]
                is_eos = gen_so_far == eos_token_id
                if is_eos.any(dim=1).all():
                    first_eos = is_eos.to(torch.int64).argmax(dim=1)
                    max_eos = first_eos.max().item()
                    x_accum = x_accum[:, : prompt_ids.size(1) + max_eos + 1]
                    break

        output = NemotronDiffusionSamplerOutput(
            sequences=x_accum,
            histories=None,
            nfe=nfe,
        )
        return output if return_dict else x_accum

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: NemotronDiffusionSamplerConfig | None = None,
        **kwargs,
    ) -> NemotronDiffusionSamplerOutput | torch.Tensor:
        del inputs, config, kwargs
        raise NotImplementedError(
            "Nemotron diffusion sampler only exposes generate/sample."
        )


def _get_num_transfer_tokens(mask_index, steps: int):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : int(remainder[i])] += 1
    return num_transfer_tokens
