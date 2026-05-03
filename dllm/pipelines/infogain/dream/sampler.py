"""
Reference: https://huggingface.co/Dream-org/Dream-v0-Base-7B/blob/main/generation_utils.py
"""

from dataclasses import dataclass

import torch
import torch.distributions as dists
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.pipelines.dream.models.generation_utils import top_k_logits, top_p_logits
from dllm.pipelines.infogain import info_gain_ops as ig


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


@dataclass
class InfoGainDreamSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 20
    max_length: int = None  # Uses prompt length + max_new_tokens when None
    steps: int = 512
    eps: float = 1e-3
    alg: str = "origin"
    alg_temp: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    stochastic_transfer: bool = False
    right_shift_logits: bool = True
    threshold: float | None = None
    use_cache: str | None = None  # None | "prefix" | "dual"
    block_size: int = 32
    candidate_number: int = 8
    position_temperature: float = 0.2
    info_gain_variant: str = "info_gain"  # "info_gain" | "lookum"


@dataclass
class InfoGainDreamSampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor] | list[list[int]],
        config: InfoGainDreamSamplerConfig | None = None,
        generation_tokens_hook_func=lambda step, x, logits: x,
        generation_logits_hook_func=lambda step, x, logits: logits,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Diffusion-style masked decoding for *generation from inputs*.
        (docstring unchanged)
        """
        config = config or InfoGainDreamSamplerConfig()

        # Pull args from config with kwargs overrides
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        steps = kwargs.get("steps", config.steps)
        eps = kwargs.get("eps", config.eps)
        alg = kwargs.get("alg", config.alg)
        alg_temp = kwargs.get("alg_temp", config.alg_temp)
        temperature = kwargs.get("temperature", config.temperature)
        top_p = kwargs.get("top_p", config.top_p)
        top_k = kwargs.get("top_k", config.top_k)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        threshold = kwargs.get("threshold", config.threshold)
        use_cache = kwargs.get("use_cache", config.use_cache)
        block_size = kwargs.get("block_size", config.block_size)
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        candidate_number = kwargs.get("candidate_number", config.candidate_number)
        position_temperature = kwargs.get(
            "position_temperature", config.position_temperature
        )
        info_gain_variant = kwargs.get(
            "info_gain_variant", config.info_gain_variant
        )

        if use_cache == "none":
            use_cache = None
        if use_cache not in (None, "prefix", "dual"):
            raise RuntimeError(
                f"Unknown use_cache mode: {use_cache}. Expected None, 'prefix', or 'dual'."
            )

        # --- Initialization ---
        mask_token_id = self.tokenizer.mask_token_id
        eos_token_id = self.tokenizer.eos_token_id

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]
        if max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + max(prompt_lens)
        elif max_new_tokens is None and max_length is not None:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length
        x = torch.full((B, T), eos_token_id, dtype=torch.long, device=self.model.device)

        seq_lens = []
        for i, p in enumerate(inputs):
            total_len = prompt_lens[i] + max_new_tokens
            seq_lens.append(total_len)
            start = T - total_len
            x[i, start : start + prompt_lens[i]] = p
            x[i, start + prompt_lens[i] : T] = mask_token_id

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for j, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[j, -L:] = 1  # Mandate to be left-padding

        if attention_mask is not None and torch.any(attention_mask == 0):
            pos_id = attention_mask.long().cumsum(-1) - 1
            pos_id.masked_fill_(attention_mask == 0, 1)
        else:
            pos_id = None

        def shift_and_hook(
            step: int | None, tokens: torch.Tensor, logits: torch.Tensor
        ):
            if right_shift_logits:
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            return generation_logits_hook_func(step, tokens, logits)

        def sample_with_alg(
            mask_logits: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            kwargs_tokens = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
            if alg in ("maskgit_plus", "confidence_threshold"):
                return sample_tokens(mask_logits, **kwargs_tokens)
            if alg == "topk_margin":
                return sample_tokens(
                    mask_logits, margin_confidence=True, **kwargs_tokens
                )
            if alg == "entropy":
                return sample_tokens(mask_logits, neg_entropy=True, **kwargs_tokens)
            raise RuntimeError(f"Unknown alg: {alg}")

        if use_cache is None:
            mask_index = x == mask_token_id
            num_transfer_tokens_list = get_num_transfer_tokens(
                mask_index=mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens_list.size(1)
            # --- Iterative refinement ---
            x = generation_tokens_hook_func(None, x, None)
            histories = [x.clone()] if return_dict else None

            if alg == "info_gain":
                if B != 1:
                    raise RuntimeError(
                        "Dream Info-Gain decoding (alg='info_gain') requires batch size 1."
                    )
                if threshold is None:
                    raise RuntimeError(
                        "Pass `threshold` for high-confidence bypass (e.g. 0.8)."
                    )
                gen_seg_start = T - seq_lens[0] + prompt_lens[0]
                gen_seg_end = T
                step_i = 0
                while True:
                    mask_index = x == mask_token_id
                    if not mask_index.any():
                        break
                    logits = self.model(x, attention_mask, pos_id).logits
                    logits = shift_and_hook(step_i, x, logits)
                    lx = logits[0:1]
                    xx = x[0:1]
                    ma = mask_index[0:1]

                    probs = F.softmax(
                        lx[:, gen_seg_start:gen_seg_end].float(), dim=-1
                    )
                    max_conf = probs.max(-1).values
                    block_masked = ma[0, gen_seg_start:gen_seg_end]
                    if (
                        threshold > 0
                        and block_masked.any()
                        and (max_conf[0][block_masked] >= threshold).all()
                    ):
                        xx_new = ig.greedy_unmask_block(
                            xx,
                            lx,
                            ma,
                            gen_seg_start,
                            gen_seg_end,
                            temperature,
                            mask_token_id,
                        )
                        x[0:1] = xx_new
                    else:
                        k = 1
                        result = ig.generate_candidates(
                            lx,
                            xx,
                            ma,
                            gen_seg_start,
                            gen_seg_end,
                            k,
                            candidate_number,
                            temperature,
                            position_temperature,
                        )
                        actions, x0s, conf_base, valid, _ = result
                        if actions is None:
                            if valid.shape[0] == 0:
                                break
                            best_pos = valid[conf_base[0, valid].argmax()].unsqueeze(0)
                            x[0, best_pos] = x0s[0][0, best_pos]
                        else:
                            nc = len(actions)
                            x_batch = xx.expand(nc, -1).clone()
                            for ii, (act, x0) in enumerate(zip(actions, x0s)):
                                x_batch[ii, act] = x0[0, act]
                            attn = attention_mask[0:1].expand(nc, -1)
                            next_logits = self.model(
                                x_batch, attn, pos_id.expand(nc, -1) if pos_id is not None else None
                            ).logits
                            if right_shift_logits:
                                next_logits = torch.cat(
                                    [next_logits[:, :1], next_logits[:, :-1]], dim=1
                                )
                            _, _, J = ig.score_candidates(
                                lx,
                                next_logits,
                                x_batch,
                                actions,
                                mask_token_id,
                                info_gain_variant,
                            )
                            best = int(J.argmax().item())
                            act = actions[best]
                            x[0, act] = x0s[best][0, act]

                    step_i += 1
                    x = generation_tokens_hook_func(step_i, x, logits)
                    if histories is not None:
                        histories.append(x.clone())

            elif alg == "confidence_threshold":
                if threshold is None:
                    raise RuntimeError(
                        "Missing `threshold` for alg == 'confidence_threshold'. "
                        "Pass it via sample(..., threshold=...)."
                    )
                i = 0
                while True:
                    mask_index = x == mask_token_id
                    logits = self.model(x, attention_mask, pos_id).logits

                    logits = shift_and_hook(i, x, logits)

                    mask_logits = logits[mask_index]
                    confidence, x0 = sample_with_alg(mask_logits)

                    full_confidence = torch.full(
                        mask_index.shape,
                        -torch.inf,
                        device=logits.device,
                        dtype=confidence.dtype,
                    )
                    full_confidence[mask_index] = confidence

                    if not torch.any(mask_index[0]):
                        continue

                    # Budget this step: top-k where k = num_transfer + leftover from past steps
                    current_transfer_tokens = (
                        int(mask_index.sum().item())
                        - num_transfer_tokens_list[0, i + 1 :].sum().item()
                        if i + 1 < effective_steps
                        else int(mask_index.sum().item())
                    )

                    selected_confidence, select_index = torch.topk(
                        full_confidence, current_transfer_tokens
                    )
                    transfer_index = torch.zeros_like(
                        mask_index, dtype=torch.bool, device=mask_index.device
                    )

                    # Start by selecting all top-k
                    transfer_index[0, select_index[0]] = True

                    # Threshold-filter within top-k (keep top-1 always, so start from 1)
                    for kk in range(1, current_transfer_tokens):
                        if selected_confidence[0, kk] < threshold:
                            transfer_index[0, select_index[0, kk]] = False

                    # Safety: never transfer unmasked positions
                    transfer_index &= mask_index

                    x_ = torch.full_like(x, mask_token_id, device=self.model.device)
                    x_[mask_index] = x0.clone()
                    x[transfer_index] = x_[transfer_index]
                    i += 1
                    x = generation_tokens_hook_func(i, x, logits)
                    if histories is not None:
                        histories.append(x.clone())

                    if not torch.any(x == mask_token_id):
                        break

            else:
                for i in range(effective_steps):
                    mask_index = x == mask_token_id

                    logits = self.model(x, attention_mask, pos_id).logits

                    logits = shift_and_hook(i, x, logits)

                    mask_logits = logits[mask_index]
                    confidence, x0 = sample_with_alg(mask_logits)

                    full_confidence = torch.full(
                        mask_index.shape,
                        -torch.inf,
                        device=logits.device,
                        dtype=confidence.dtype,
                    )
                    full_confidence[mask_index] = confidence

                    for j in range(full_confidence.shape[0]):
                        number_transfer_tokens = num_transfer_tokens_list[j, i]
                        if number_transfer_tokens > 0:
                            if alg_temp is None or alg_temp == 0:
                                _, transfer_index = torch.topk(
                                    full_confidence[j], number_transfer_tokens
                                )
                            else:
                                fc = full_confidence[j] / alg_temp
                                fc = F.softmax(fc, dim=-1)
                                transfer_index = torch.multinomial(
                                    fc, num_samples=number_transfer_tokens
                                )

                            x_ = torch.full_like(
                                x, mask_token_id, device=self.model.device
                            )
                            x_[mask_index] = x0.clone()
                            x[j, transfer_index] = x_[j, transfer_index]

                    x = generation_tokens_hook_func(i, x, logits)
                    if histories is not None:
                        histories.append(x.clone())

            if not return_dict:
                return x
            else:
                return BaseSamplerOutput(sequences=x, histories=histories)

        else:
            dual_cache = use_cache == "dual"

            gen_length = max_new_tokens
            if block_size is None:
                block_size = gen_length
            assert gen_length % block_size == 0, (
                f"gen_length ({gen_length}) must be divisible by block_size "
                f"({block_size})"
            )
            num_blocks = gen_length // block_size

            assert (
                steps % num_blocks == 0
            ), f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
            steps_per_block = steps // num_blocks
            timesteps = torch.linspace(1, eps, steps_per_block + 1, device=x.device)
            if attention_mask is not None and torch.any(attention_mask == 0):
                cache_attention_mask = torch.logical_and(
                    attention_mask.bool().unsqueeze(1).unsqueeze(-2),
                    attention_mask.bool().unsqueeze(1).unsqueeze(-1),
                )
                tok_idx = pos_id
            else:
                cache_attention_mask = "full"
                tok_idx = None

            x = generation_tokens_hook_func(None, x, None)
            histories = [x.clone()] if return_dict else None
            global_step = 0

            gen_start = T - max_new_tokens  # == max(prompt_lens)

            past_key_values = None

            for num_block in range(num_blocks):
                current_block_start = gen_start + num_block * block_size
                current_block_end = current_block_start + block_size

                # update cache
                model_output = self.model(
                    x, cache_attention_mask, tok_idx, use_cache=True
                )
                past_key_values = model_output.past_key_values
                logits = shift_and_hook(global_step, x, model_output.logits)

                _, x0_full = sample_tokens(
                    logits, temperature=temperature, top_p=top_p, top_k=top_k
                )
                x[:, current_block_start] = x0_full[:, current_block_start]

                x = generation_tokens_hook_func(global_step, x, logits)
                if histories is not None:
                    histories.append(x.clone())
                global_step += 1

                replace_position = None
                if not dual_cache:
                    new_past_key_values = []
                    for li in range(len(past_key_values)):
                        new_past_key_values.append(())
                        for kj in range(len(past_key_values[li])):
                            new_past_key_values[li] += (
                                past_key_values[li][kj][:, :current_block_start, :],
                            )
                    past_key_values = new_past_key_values
                else:
                    replace_position = torch.zeros_like(x, dtype=torch.bool)
                    replace_position[:, current_block_start:current_block_end] = True

                inner_step = 1
                while True:
                    end = current_block_end if dual_cache else None
                    region = x[:, current_block_start:end]

                    mask_index = region == mask_token_id
                    mask_index[:, block_size:] = False

                    if cache_attention_mask != "full":
                        current_attention_mask = cache_attention_mask[
                            :, :, :, current_block_start:
                        ]
                    else:
                        current_attention_mask = cache_attention_mask

                    region_tok_idx = (
                        tok_idx[:, current_block_start:end]
                        if tok_idx is not None
                        else None
                    )

                    model_output = self.model(
                        region,
                        current_attention_mask,
                        region_tok_idx,
                        past_key_values=past_key_values,
                        use_cache=True,
                        dual_cache=dual_cache,
                        replace_position=replace_position,
                    )
                    logits = shift_and_hook(global_step, x, model_output.logits)
                    mask_logits = logits[mask_index]

                    confidence, x0 = sample_with_alg(mask_logits)

                    current_transfer_tokens = (
                        x[:, current_block_start:current_block_end] == mask_token_id
                    ).sum()

                    full_confidence = torch.full_like(
                        region,
                        -torch.inf,
                        device=self.model.device,
                        dtype=logits.dtype,
                    )
                    full_confidence[mask_index] = confidence
                    full_confidence[:, block_size:] = -torch.inf
                    x_ = torch.full_like(
                        region, mask_token_id, device=self.model.device
                    )
                    x_[mask_index] = x0.clone()

                    if alg == "confidence_threshold":
                        selected_confidence, select_index = torch.topk(
                            full_confidence, current_transfer_tokens
                        )
                        transfer_index = torch.zeros_like(
                            x_, device=x.device, dtype=torch.bool
                        )

                        select_index = select_index.to(x.device)
                        transfer_index[0, select_index[0]] = True
                        for k in range(1, current_transfer_tokens):
                            if selected_confidence[0, k] < threshold:
                                transfer_index[0, select_index[0, k]] = False
                        x[:, current_block_start:end][transfer_index] = x_[
                            transfer_index
                        ]

                    else:
                        if inner_step == steps_per_block:
                            break
                        t = timesteps[inner_step]
                        s = timesteps[inner_step + 1]
                        num_mask_token = mask_index.sum() / mask_index.shape[0]

                        number_transfer_tokens = (
                            int(num_mask_token * (1 - s / t))
                            if inner_step < steps_per_block - 1
                            else int(num_mask_token)
                        )
                        if number_transfer_tokens > 0:
                            if alg_temp is None or alg_temp == 0:
                                _, select_index = torch.topk(
                                    full_confidence, number_transfer_tokens
                                )
                            else:
                                fc = full_confidence / alg_temp
                                fc = F.softmax(fc, dim=-1)
                                select_index = torch.multinomial(
                                    fc, num_samples=number_transfer_tokens
                                )

                            transfer_index = torch.zeros_like(
                                x_, device=x.device, dtype=torch.bool
                            )
                            transfer_index.scatter_(1, select_index, True)

                            transfer_index &= mask_index
                            x[:, current_block_start:end][transfer_index] = x_[
                                transfer_index
                            ]

                            x = generation_tokens_hook_func(global_step, x, logits)
                    if histories is not None:
                        histories.append(x.clone())
                    global_step += 1

                    inner_step += 1
                    if (
                        x[:, current_block_start:current_block_end] == mask_token_id
                    ).sum() == 0:
                        break

            if not return_dict:
                return x
            else:
                return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor] | list[list[int]],
        config: InfoGainDreamSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput:
        raise NotImplementedError
