"""
DiffusionGemma sampler for the dllm sampler interface.

Example:
    python -u examples/diffusiongemma/sample.py \
        --model_name_or_path /mnt/weka/shrd/research/model/diffusiongemma-26B-A4B-it
"""

import math
from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput


@dataclass
class DiffusionGemmaSamplerOutput(BaseSamplerOutput):
    tokens_per_forward: torch.Tensor | None = None
    decoder_forward_passes: torch.Tensor | None = None


@dataclass
class DiffusionGemmaSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 256
    steps: int = 48
    entropy_threshold: float = 0.005
    stability_threshold: int = 1
    entropy_bound: float = 0.1
    max_temperature: float = 0.8
    min_temperature: float = 0.4
    pad_token_id: int | None = None
    eos_token_id: int | list[int] | None = None


@dataclass
class DiffusionGemmaSampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list] | torch.Tensor,
        config: DiffusionGemmaSamplerConfig | None = None,
        **kwargs,
    ) -> DiffusionGemmaSamplerOutput | torch.Tensor:
        if config is None:
            config = DiffusionGemmaSamplerConfig()
        return_dict = bool(kwargs.pop("return_dict", config.return_dict))

        model_config = getattr(self.model, "config", None)
        text_config = model_config.text_config
        canvas_length = model_config.canvas_length
        encoder = self.model.model.encoder
        decoder = self.model.model.decoder
        decoder_dtype = decoder.embed_tokens.weight.dtype
        device = torch.device(self.model.device)
        pad_token_id = config.pad_token_id
        if pad_token_id is None and self.tokenizer.pad_token_id is not None:
            pad_token_id = int(self.tokenizer.pad_token_id)
        eos_token_id = config.eos_token_id
        if eos_token_id is None:
            eos_token_id = getattr(model_config, "eos_token_id", None)

        if isinstance(inputs, torch.Tensor):
            prompts = [inputs] if inputs.dim() == 1 else [row for row in inputs]
        elif inputs and all(isinstance(token_id, int) for token_id in inputs):
            prompts = [inputs]
        else:
            prompts = list(inputs)
        if not prompts:
            raise ValueError("DiffusionGemmaSampler requires at least one input prompt.")

        tensors = [
            prompt.to(device=device, dtype=torch.long)
            if isinstance(prompt, torch.Tensor)
            else torch.as_tensor(prompt, dtype=torch.long, device=device)
            for prompt in prompts
        ]
        max_prompt_len = max(tensor.numel() for tensor in tensors)
        input_ids = torch.full(
            (len(tensors), max_prompt_len),
            fill_value=int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, tensor in enumerate(tensors):
            input_ids[i, : tensor.numel()] = tensor.reshape(-1)
            attention_mask[i, : tensor.numel()] = True

        batch_size = input_ids.shape[0]
        max_new_tokens = config.max_new_tokens
        attention_mask = kwargs.pop("attention_mask", attention_mask).to(device=device)

        if len(attention_mask.shape) > 2:
            raise ValueError("`attention_mask` passed to `sample` must be 2D.")
        attention_mask = attention_mask.bool()

        cur_len = input_ids.shape[1]
        initial_input_ids_len = cur_len
        max_new_canvases = math.ceil(max_new_tokens / canvas_length)

        device = input_ids.device
        eos_tensor = None
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
        decoder_forward_passes = torch.zeros(batch_size, dtype=torch.int, device=device)
        if eos_token_id is not None:
            eos_tensor = torch.tensor(eos_token_id, device=input_ids.device)

        cache_kwargs = {}
        if hasattr(model_config, "get_text_config"):
            cache_kwargs["config"] = model_config
        past_key_values = DynamicCache(**cache_kwargs)

        encoder_position_ids = torch.arange(
            cur_len - input_ids.shape[1],
            cur_len,
            dtype=torch.int32,
            device=input_ids.device,
        ).unsqueeze(0)
        decoder_position_ids = torch.arange(
            cur_len,
            cur_len + canvas_length,
            dtype=torch.int32,
            device=input_ids.device,
        ).unsqueeze(0)

        entropy_bound = float(config.entropy_bound)

        vocab_size = int(text_config.vocab_size)
        confidence_threshold = config.entropy_threshold

        decoder_attention_mask = torch.nn.functional.pad(
            attention_mask, (0, canvas_length), value=True
        )

        is_prefill = True
        for _ in range(max_new_canvases):
            encoder_input_ids = input_ids if is_prefill else input_ids[:, -canvas_length:]
            encoder_input_ids = encoder_input_ids.clone(memory_format=torch.contiguous_format)
            create_encoder_masks = getattr(encoder, "create_masks_for_generate", None)
            if create_encoder_masks is None:
                encoder_attention_mask = attention_mask
            else:
                dummy_input_embeds = torch.empty(
                    (encoder_input_ids.shape[0], encoder_input_ids.shape[1], 0),
                    dtype=text_config.dtype,
                    device=encoder_input_ids.device,
                )
                encoder_attention_mask = create_encoder_masks(
                    config=model_config,
                    inputs_embeds=dummy_input_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=encoder_position_ids,
                )
            encoder_outputs = encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                position_ids=encoder_position_ids,
            )
            past_key_values = encoder_outputs.past_key_values
            is_prefill = False

            current_canvas = torch.randint(
                low=0,
                high=vocab_size,
                size=(batch_size, canvas_length),
                device=device,
            )
            self_conditioning_logits = None

            create_decoder_mask = getattr(decoder, "create_diffusion_decoder_attention_mask", None)
            mask_mapping = (
                decoder_attention_mask
                if create_decoder_mask is None
                else create_decoder_mask(
                    config=text_config,
                    inputs_embeds=current_canvas.unsqueeze(-1),
                    past_key_values=past_key_values,
                    decoder_attention_mask=decoder_attention_mask,
                )
            )
            finished_denoising = torch.zeros(batch_size, dtype=torch.bool, device=device)
            argmax_canvas_history = None
            argmax_canvas = current_canvas

            for cur_step in reversed(range(1, config.steps + 1)):
                decoder_forward_passes += ~(finished_denoising | finished_sequences)

                decoder_outputs = self.model.forward(
                    decoder_input_ids=current_canvas,
                    self_conditioning_logits=self_conditioning_logits,
                    decoder_attention_mask=mask_mapping,
                    past_key_values=past_key_values,
                    decoder_position_ids=decoder_position_ids,
                )
                raw_logits = decoder_outputs.logits
                cur_step_tensor = torch.tensor(cur_step, device=current_canvas.device, dtype=torch.int32)
                processed_logits = raw_logits
                if config.min_temperature is not None and config.max_temperature is not None:
                    temperature = config.min_temperature + (
                        (config.max_temperature - config.min_temperature)
                        * (cur_step_tensor / config.steps)
                    )
                    processed_logits = processed_logits / temperature
                probs = torch.softmax(processed_logits, dim=-1, dtype=torch.float32)

                denoiser_canvas = torch.multinomial(probs.view(-1, vocab_size), num_samples=1)
                denoiser_canvas = denoiser_canvas.squeeze(-1).view(batch_size, canvas_length)
                new_argmax_canvas = torch.argmax(processed_logits, dim=-1)

                dist = torch.distributions.Categorical(logits=processed_logits)
                token_entropy = dist.entropy()
                sorted_token_entropy, sorted_indices = torch.sort(token_entropy, dim=-1, descending=False)
                cumulative_entropy = torch.cumsum(sorted_token_entropy, dim=-1)
                sorted_selection_mask = cumulative_entropy - sorted_token_entropy <= entropy_bound
                accepted_token_mask = torch.scatter(
                    input=torch.zeros_like(sorted_selection_mask),
                    dim=-1,
                    index=sorted_indices,
                    src=sorted_selection_mask,
                )
                accepted_canvas = torch.where(accepted_token_mask, denoiser_canvas, current_canvas).clone()
                random_canvas = torch.randint(
                    low=0,
                    high=vocab_size,
                    size=accepted_canvas.shape,
                    device=accepted_canvas.device,
                )
                new_current_canvas = torch.where(~accepted_token_mask, random_canvas, accepted_canvas).clone()

                if config.stability_threshold is not None and confidence_threshold is not None:
                    (
                        new_argmax_canvas,
                        new_current_canvas,
                        processed_logits,
                        finished_denoising,
                        argmax_canvas_history,
                    ) = self._apply_adaptive_stopping(
                        finished_denoising=finished_denoising,
                        current_canvas=current_canvas,
                        new_current_canvas=new_current_canvas,
                        argmax_canvas=argmax_canvas,
                        new_argmax_canvas=new_argmax_canvas,
                        processed_logits=processed_logits,
                        self_conditioning_logits=self_conditioning_logits,
                        argmax_canvas_history=argmax_canvas_history,
                        token_entropy=token_entropy,
                        stability_threshold=config.stability_threshold,
                        confidence_threshold=confidence_threshold,
                    )

                current_canvas = new_current_canvas
                argmax_canvas = new_argmax_canvas
                self_conditioning_logits = processed_logits.to(decoder_dtype)

                if torch.all(finished_denoising):
                    break

            input_ids = torch.cat([input_ids, argmax_canvas], dim=-1)
            finished_this_canvas = torch.zeros_like(finished_sequences)
            if eos_tensor is not None:
                finished_this_canvas |= torch.isin(input_ids[:, -canvas_length:], eos_tensor).any(dim=-1)

            previously_finished_sequences = finished_sequences
            finished_sequences = previously_finished_sequences | finished_this_canvas
            if pad_token_id is not None and torch.any(finished_sequences):
                input_ids[previously_finished_sequences, -canvas_length:] = pad_token_id
                if eos_tensor is not None and torch.any(finished_this_canvas):
                    new_tokens = input_ids[:, -canvas_length:]
                    is_eos = torch.isin(new_tokens, eos_tensor)
                    eos_cumsum = is_eos.cumsum(dim=-1)
                    pad_mask = (eos_cumsum > 0) & ~((eos_cumsum == 1) & is_eos)
                    new_tokens[pad_mask] = pad_token_id

            if torch.all(finished_sequences):
                break

            cur_len += canvas_length
            attention_mask = torch.nn.functional.pad(attention_mask, (0, canvas_length), value=True)
            decoder_attention_mask = torch.nn.functional.pad(decoder_attention_mask, (0, canvas_length), value=True)
            encoder_position_ids = decoder_position_ids
            decoder_position_ids = torch.arange(
                cur_len,
                cur_len + canvas_length,
                dtype=torch.int32,
                device=decoder_position_ids.device,
            ).unsqueeze(0)

        new_tokens = input_ids[:, initial_input_ids_len:]
        if pad_token_id is not None:
            num_valid_tokens = (new_tokens != pad_token_id).sum(dim=-1)
        else:
            num_valid_tokens = torch.full(
                (input_ids.shape[0],),
                new_tokens.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            )
        tokens_per_forward = num_valid_tokens / decoder_forward_passes.clamp_min(1)
        output = DiffusionGemmaSamplerOutput(
            sequences=input_ids,
            histories=None,
            tokens_per_forward=tokens_per_forward,
            decoder_forward_passes=decoder_forward_passes,
        )
        return output if return_dict else input_ids

    @staticmethod
    def _apply_adaptive_stopping(
        *,
        finished_denoising: torch.Tensor,
        current_canvas: torch.Tensor,
        new_current_canvas: torch.Tensor,
        argmax_canvas: torch.Tensor,
        new_argmax_canvas: torch.Tensor,
        processed_logits: torch.Tensor,
        self_conditioning_logits: torch.Tensor | None,
        argmax_canvas_history: torch.Tensor | None,
        token_entropy: torch.Tensor,
        stability_threshold: int,
        confidence_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if finished_denoising.any():
            new_argmax_canvas = torch.where(finished_denoising[:, None], argmax_canvas, new_argmax_canvas)
            new_current_canvas = torch.where(finished_denoising[:, None], current_canvas, new_current_canvas)
            processed_logits = torch.where(
                finished_denoising[:, None, None],
                self_conditioning_logits,
                processed_logits,
            )

        if stability_threshold == 0:
            stable = torch.ones(
                (processed_logits.shape[0]),
                device=processed_logits.device,
                dtype=torch.bool,
            )
        else:
            if argmax_canvas_history is None:
                argmax_canvas_history = torch.full(
                    (
                        stability_threshold,
                        new_argmax_canvas.shape[0],
                        new_argmax_canvas.shape[1],
                    ),
                    -1,
                    dtype=new_argmax_canvas.dtype,
                    device=new_argmax_canvas.device,
                )
            stable = (argmax_canvas_history == new_argmax_canvas[None, :, :]).all(dim=-1).all(dim=0)
            argmax_canvas_history = torch.roll(argmax_canvas_history, shifts=-1, dims=0)
            argmax_canvas_history[-1] = new_argmax_canvas

        if finished_denoising.any():
            confident = torch.mean(torch.distributions.Categorical(logits=processed_logits).entropy(), dim=-1)
        else:
            confident = torch.mean(token_entropy, dim=-1)
        finished_denoising |= stable & (confident < confidence_threshold)

        return (
            new_argmax_canvas,
            new_current_canvas,
            processed_logits,
            finished_denoising,
            argmax_canvas_history,
        )

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: DiffusionGemmaSamplerConfig | None = None,
        **kwargs,
    ) -> DiffusionGemmaSamplerOutput | torch.Tensor:
        del inputs, config, kwargs
        raise NotImplementedError("DiffusionGemma native generator does not expose infill in dllm yet.")
