"""
DiffusionGemma sampler for the dllm sampler interface.

Example:
    python -u examples/diffusiongemma/sample.py \
        --model_name_or_path /mnt/weka/shrd/research/model/diffusiongemma-26B-A4B-it
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import torch

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput


@dataclass
class DiffusionGemmaSamplerOutput(BaseSamplerOutput):
    tokens_per_forward: torch.Tensor | None = None
    decoder_forward_passes: torch.Tensor | None = None
    past_key_values: Any | None = None
    logits: Any | None = None
    scores: Any | None = None
    raw_output: Any | None = None


@dataclass
class DiffusionGemmaSamplerConfig(BaseSamplerConfig):
    # Length / block config
    max_new_tokens: int = 256
    max_length: int | None = None
    block_size: int = 256

    # Diffusion denoising config
    max_denoising_steps: int = 48
    entropy_threshold: float = 0.005
    stability_threshold: int = 1

    # Entropy sampler config
    entropy_bound: float = 0.1

    # Temperature config
    max_temperature: float = 0.8
    min_temperature: float = 0.4

    # Runtime generation defaults
    cache_implementation: str | None = None
    cache_config: Any | None = None
    bos_token_id: int | None = None
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

        config_values, config_overrides = self._resolve_config(config, kwargs)
        return_dict = config_values.return_dict
        model_kwargs = dict(kwargs)
        generation_config = model_kwargs.pop("generation_config", None)

        device = torch.device(self.model.device)
        input_ids, attention_mask, prompt_lens = self._normalize_inputs(inputs, device)
        batch_size = input_ids.shape[0]
        if batch_size == 0:
            raise ValueError("DiffusionGemmaSampler requires at least one input prompt.")

        max_new_tokens = config_values.max_new_tokens
        if config_values.max_length is not None and "max_new_tokens" not in config_overrides:
            max_new_tokens = config_values.max_length - max(prompt_lens)
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive after accounting for the prompt length.")

        canvas_length = getattr(getattr(self.model, "config", None), "canvas_length", None)
        if canvas_length is not None and config_values.block_size != canvas_length:
            raise ValueError(
                "DiffusionGemma native generation uses model.config.canvas_length. "
                f"Got block_size={config_values.block_size}, model canvas_length={canvas_length}."
            )

        generation_config = self._build_generation_config(
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            config_values=config_values,
        )

        attention_mask = model_kwargs.pop("attention_mask", attention_mask).to(device=device)
        past_key_values = model_kwargs.pop("past_key_values", None)
        streamer = model_kwargs.pop("streamer", None)
        unsupported_generation_kwargs = {
            "logits_processor",
            "stopping_criteria",
            "sampler_config",
            "t_min",
            "t_max",
            "confidence_threshold",
        }
        unexpected_generation_kwargs = sorted(unsupported_generation_kwargs & model_kwargs.keys())
        if unexpected_generation_kwargs:
            raise ValueError(
                "DiffusionGemmaSampler does not accept HF generation kwargs directly: "
                f"{unexpected_generation_kwargs}. Use DiffusionGemmaSamplerConfig or pass a generation_config."
            )

        output = self._sample_loop(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            past_key_values=past_key_values,
            streamer=streamer,
            **model_kwargs,
        )
        sequences = output.sequences if hasattr(output, "sequences") else output
        if not return_dict:
            return sequences

        return DiffusionGemmaSamplerOutput(
            sequences=sequences,
            histories=None,
            tokens_per_forward=getattr(output, "tokens_per_forward", None),
            decoder_forward_passes=getattr(output, "decoder_forward_passes", None),
            past_key_values=getattr(output, "past_key_values", None),
            logits=getattr(output, "logits", None),
            scores=getattr(output, "scores", None),
            raw_output=output,
        )

    def _sample_loop(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        generation_config: Any,
        past_key_values: Any | None = None,
        streamer: Any | None = None,
        **model_kwargs,
    ) -> Any:
        batch_size, cur_len = input_ids.shape
        initial_input_ids_len = cur_len
        if past_key_values is not None:
            cur_len += past_key_values.get_seq_length()
        max_new_tokens = generation_config.max_new_tokens
        max_length = max_new_tokens + cur_len
        max_new_canvases = math.ceil(max_new_tokens / self.model.config.canvas_length)

        if past_key_values is not None and generation_config.cache_implementation is not None:
            raise ValueError(
                "Cannot provide both `past_key_values` and "
                "`generation_config.cache_implementation`."
            )
        if (
            "pixel_values" not in model_kwargs
            and input_ids is not None
            and hasattr(self.model.config, "image_token_id")
            and (input_ids == self.model.config.image_token_id).any()
        ):
            # Match the native warning path without requiring the HF logger here.
            import warnings

            warnings.warn(
                "Your input tokens contain image tokens, but you haven't set `pixel_values`.",
                stacklevel=2,
            )

        device = input_ids.device
        canvas_length = self.model.config.canvas_length
        eos_tensor = None
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
        decoder_forward_passes = torch.zeros(batch_size, dtype=torch.int, device=device)
        if past_key_values is None:
            try:
                from transformers.cache_utils import DynamicCache, QuantizedCache, StaticCache
            except ImportError as error:
                raise ImportError(
                    "DiffusionGemmaSampler requires transformers cache classes for generation."
                ) from error

            cache_implementation = getattr(generation_config, "cache_implementation", None)
            model_config = getattr(self.model, "config", None)
            cache_config = model_config if hasattr(model_config, "get_text_config") else None
            layer_config = cache_config or self.model.config.text_config
            cache_max_length = max_length - canvas_length

            if cache_implementation in {"static", "offloaded_static", "sliding_window", "hybrid"}:
                offloading = "offloaded" in cache_implementation
                cache_to_check = getattr(self, "_cache", None)
                need_new_cache = (
                    cache_to_check is None
                    or not isinstance(cache_to_check, StaticCache)
                    or getattr(cache_to_check, "offloading", None) != offloading
                    or getattr(cache_to_check, "max_batch_size", None) != batch_size
                    or getattr(cache_to_check, "max_cache_len", 0) < cache_max_length
                )
                if need_new_cache:
                    self._cache = StaticCache(
                        config=layer_config,
                        max_cache_len=cache_max_length,
                        offloading=offloading,
                    )
                else:
                    self._cache.reset()
                past_key_values = self._cache
            elif cache_implementation == "quantized":
                cache_config = dict(getattr(generation_config, "cache_config", None) or {})
                if layer_config is not None:
                    cache_config.setdefault("config", layer_config)
                backend = cache_config.pop("backend", "quanto")
                past_key_values = QuantizedCache(backend=backend, **cache_config)
            else:
                dynamic_cache_kwargs = {}
                if cache_implementation != "dynamic_full" and cache_config is not None:
                    dynamic_cache_kwargs["config"] = cache_config
                if cache_implementation == "offloaded":
                    dynamic_cache_kwargs["offloading"] = True
                past_key_values = DynamicCache(**dynamic_cache_kwargs)
        if generation_config.eos_token_id is not None:
            eos_tensor = torch.tensor(generation_config.eos_token_id, device=input_ids.device)

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

        if len(attention_mask.shape) > 2:
            raise ValueError("`attention_mask` passed to `sample` must be 2D.")
        attention_mask = attention_mask.bool()

        entropy_bound = float(generation_config.entropy_bound)

        vocab_size = int(self.model.config.text_config.vocab_size)
        min_temperature = generation_config.min_temperature
        max_temperature = generation_config.max_temperature
        stability_threshold = generation_config.stability_threshold
        confidence_threshold = generation_config.entropy_threshold
        if streamer is not None:
            streamer.put(input_ids.cpu())

        decoder_attention_mask = torch.nn.functional.pad(
            attention_mask, (0, canvas_length), value=True
        )

        is_prefill = True
        for _ in range(max_new_canvases):
            unprocessed_input_ids = input_ids if is_prefill else input_ids[:, -canvas_length:]
            unprocessed_input_ids = unprocessed_input_ids.clone(memory_format=torch.contiguous_format)
            create_encoder_masks = getattr(self.model.model.encoder, "create_masks_for_generate", None)
            if create_encoder_masks is None:
                encoder_mask_mapping = attention_mask
            else:
                dummy_input_embeds = torch.empty(
                    (batch_size, unprocessed_input_ids.shape[1], 0),
                    dtype=self.model.config.text_config.dtype,
                    device=input_ids.device,
                )
                encoder_mask_mapping = create_encoder_masks(
                    config=self.model.config,
                    inputs_embeds=dummy_input_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=encoder_position_ids,
                    mm_token_type_ids=model_kwargs.get("mm_token_type_ids"),
                )
            encoder_outputs = self.model.model.encoder(
                input_ids=unprocessed_input_ids,
                attention_mask=encoder_mask_mapping,
                past_key_values=past_key_values,
                position_ids=encoder_position_ids,
                **model_kwargs,
            )
            past_key_values = encoder_outputs.past_key_values
            is_prefill = False

            for key in ("pixel_values", "image_position_ids", "mm_token_type_ids"):
                model_kwargs.pop(key, None)

            current_canvas = model_kwargs.pop("decoder_input_ids", None)
            if current_canvas is None:
                current_canvas = torch.randint(
                    low=0,
                    high=vocab_size,
                    size=(batch_size, canvas_length),
                    device=device,
                )
            self_conditioning_logits = model_kwargs.pop("self_conditioning_logits", None)

            create_decoder_mask = getattr(self.model.model.decoder, "create_diffusion_decoder_attention_mask", None)
            if create_decoder_mask is None:
                mask_mapping = decoder_attention_mask
            else:
                mask_mapping = create_decoder_mask(
                    config=self.model.config.text_config,
                    inputs_embeds=current_canvas.unsqueeze(-1),
                    past_key_values=past_key_values,
                    decoder_attention_mask=decoder_attention_mask,
                )
            finished_denoising = torch.zeros(batch_size, dtype=torch.bool, device=device)
            argmax_canvas_history = None
            argmax_canvas = current_canvas

            for cur_step in reversed(range(1, generation_config.max_denoising_steps + 1)):
                decoder_forward_passes += ~(finished_denoising | finished_sequences)

                current_canvas, argmax_canvas, self_conditioning_logits, finished_denoising, argmax_canvas_history = (
                    self._denoising_step(
                        current_canvas=current_canvas,
                        argmax_canvas=argmax_canvas,
                        input_ids=input_ids,
                        decoder_position_ids=decoder_position_ids,
                        self_conditioning_logits=self_conditioning_logits,
                        mask_mapping=mask_mapping,
                        past_key_values=past_key_values,
                        finished_denoising=finished_denoising,
                        cur_step=cur_step,
                        vocab_size=vocab_size,
                        entropy_bound=entropy_bound,
                        min_temperature=min_temperature,
                        max_temperature=max_temperature,
                        max_denoising_steps=generation_config.max_denoising_steps,
                        stability_threshold=stability_threshold,
                        confidence_threshold=confidence_threshold,
                        argmax_canvas_history=argmax_canvas_history,
                        **model_kwargs,
                    )
                )

                if streamer is not None and hasattr(streamer, "put_draft"):
                    streamer_kwargs = {"value": argmax_canvas.cpu()}
                    if getattr(streamer, "_takes_logits", False):
                        streamer_kwargs = {"logits": self_conditioning_logits.cpu()}
                    streamer.put_draft(**streamer_kwargs)

                if torch.all(finished_denoising):
                    break

            input_ids = torch.cat([input_ids, argmax_canvas], dim=-1)
            finished_this_canvas = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
            if getattr(generation_config, "max_length", None) is not None:
                finished_this_canvas |= input_ids.shape[-1] >= generation_config.max_length
            if eos_tensor is not None:
                finished_this_canvas |= torch.isin(input_ids[:, -canvas_length:], eos_tensor).any(dim=-1)
            previously_finished_sequences = finished_sequences
            finished_sequences = previously_finished_sequences | finished_this_canvas
            if generation_config.pad_token_id is not None and torch.any(finished_sequences):
                input_ids[previously_finished_sequences, -canvas_length:] = generation_config.pad_token_id
                if eos_tensor is not None and torch.any(finished_this_canvas):
                    new_tokens = input_ids[:, -canvas_length:]
                    is_eos = torch.isin(new_tokens, eos_tensor)
                    eos_cumsum = is_eos.cumsum(dim=-1)
                    pad_mask = (eos_cumsum > 0) & ~((eos_cumsum == 1) & is_eos)
                    new_tokens[pad_mask] = generation_config.pad_token_id

            if streamer is not None:
                streamer.put(input_ids[:, -canvas_length:].cpu())

            if torch.all(finished_sequences):
                break

            cur_len += canvas_length
            decoder_attention_mask = torch.nn.functional.pad(decoder_attention_mask, (0, canvas_length), value=True)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, canvas_length), value=True)
            encoder_position_ids = decoder_position_ids
            decoder_position_ids = torch.arange(
                cur_len,
                cur_len + canvas_length,
                dtype=torch.int32,
                device=decoder_position_ids.device,
            ).unsqueeze(0)

        if streamer is not None:
            streamer.end()

        tokens_per_forward = self._compute_tokens_per_forward(
            input_ids,
            decoder_forward_passes,
            initial_input_ids_len,
            generation_config.pad_token_id,
        )
        return SimpleNamespace(
            sequences=input_ids,
            tokens_per_forward=tokens_per_forward,
            decoder_forward_passes=decoder_forward_passes,
            past_key_values=past_key_values,
            logits=None,
            scores=None,
        )

    def _denoising_step(
        self,
        current_canvas: torch.Tensor,
        argmax_canvas: torch.Tensor,
        input_ids: torch.Tensor,
        decoder_position_ids: torch.Tensor,
        self_conditioning_logits: torch.Tensor | None,
        mask_mapping: Any,
        past_key_values: Any,
        finished_denoising: torch.Tensor,
        cur_step: int,
        vocab_size: int,
        entropy_bound: float,
        min_temperature: float | None,
        max_temperature: float | None,
        max_denoising_steps: int,
        stability_threshold: int | None,
        confidence_threshold: float | None,
        argmax_canvas_history: torch.Tensor | None,
        **model_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        decoder_outputs = self.model.forward(
            decoder_input_ids=current_canvas,
            self_conditioning_logits=self_conditioning_logits,
            decoder_attention_mask=mask_mapping,
            past_key_values=past_key_values,
            decoder_position_ids=decoder_position_ids,
            **model_kwargs,
        )
        raw_logits = decoder_outputs.logits
        cur_step_tensor = torch.tensor(cur_step, device=current_canvas.device, dtype=torch.int32)
        processed_logits = raw_logits
        if min_temperature is not None and max_temperature is not None:
            temperature = min_temperature + ((max_temperature - min_temperature) * (cur_step_tensor / max_denoising_steps))
            processed_logits = processed_logits / temperature
        probs = torch.softmax(processed_logits, dim=-1, dtype=torch.float32)

        batch_size, canvas_length = current_canvas.shape
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

        if stability_threshold is not None and confidence_threshold is not None:
            if finished_denoising.any():
                new_argmax_canvas = torch.where(finished_denoising[:, None], argmax_canvas, new_argmax_canvas)
                new_current_canvas = torch.where(finished_denoising[:, None], current_canvas, new_current_canvas)
                processed_logits = torch.where(
                    finished_denoising[:, None, None],
                    self_conditioning_logits,
                    processed_logits,
                )

            if stability_threshold == 0:
                stable = torch.ones((processed_logits.shape[0]), device=processed_logits.device, dtype=torch.bool)
            else:
                if argmax_canvas_history is None:
                    argmax_canvas_history = torch.full(
                        (stability_threshold, new_argmax_canvas.shape[0], new_argmax_canvas.shape[1]),
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

        self_conditioning_logits = processed_logits.to(self.model.model.decoder.embed_tokens.weight.dtype)
        return new_current_canvas, new_argmax_canvas, self_conditioning_logits, finished_denoising, argmax_canvas_history

    @staticmethod
    def _compute_tokens_per_forward(
        input_ids: torch.Tensor,
        decoder_forward_passes: torch.Tensor,
        initial_input_ids_len: int,
        pad_token_id: int | None,
    ) -> torch.Tensor:
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
        return num_valid_tokens / decoder_forward_passes.clamp_min(1)

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: DiffusionGemmaSamplerConfig | None = None,
        **kwargs,
    ) -> DiffusionGemmaSamplerOutput | torch.Tensor:
        del inputs, config, kwargs
        raise NotImplementedError("DiffusionGemma native generator does not expose infill in dllm yet.")

    @staticmethod
    def _resolve_config(
        config: DiffusionGemmaSamplerConfig,
        kwargs: dict[str, Any],
    ) -> tuple[DiffusionGemmaSamplerConfig, set[str]]:
        field_names = DiffusionGemmaSamplerConfig.__dataclass_fields__.keys()
        overrides = {name: kwargs.pop(name) for name in list(kwargs) if name in field_names}
        return replace(config, **overrides), set(overrides)

    def _normalize_inputs(
        self,
        inputs: list[torch.Tensor | list] | torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
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
        prompt_lens = [tensor.numel() for tensor in tensors]
        max_prompt_len = max(prompt_lens)
        pad_id = int(self.tokenizer.pad_token_id)

        input_ids = torch.full(
            (len(tensors), max_prompt_len),
            fill_value=pad_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, tensor in enumerate(tensors):
            input_ids[i, : tensor.numel()] = tensor.reshape(-1)
            attention_mask[i, : tensor.numel()] = True
        return input_ids, attention_mask, prompt_lens

    def _build_generation_config(
        self,
        generation_config: Any | None,
        max_new_tokens: int,
        config_values: DiffusionGemmaSamplerConfig,
    ) -> DiffusionGemmaSamplerConfig:
        model_generation_config = copy.deepcopy(generation_config or getattr(self.model, "generation_config", None))
        return replace(
            config_values,
            max_new_tokens=max_new_tokens,
            cache_implementation=config_values.cache_implementation
            or getattr(model_generation_config, "cache_implementation", None),
            cache_config=config_values.cache_config or getattr(model_generation_config, "cache_config", None),
            bos_token_id=config_values.bos_token_id
            or getattr(model_generation_config, "bos_token_id", self.model.config.text_config.bos_token_id),
            pad_token_id=config_values.pad_token_id
            if config_values.pad_token_id is not None
            else (
                getattr(model_generation_config, "pad_token_id", None)
                if getattr(model_generation_config, "pad_token_id", None) is not None
                else int(self.tokenizer.pad_token_id)
            ),
            eos_token_id=config_values.eos_token_id
            if config_values.eos_token_id is not None
            else (
                getattr(model_generation_config, "eos_token_id", None)
                if getattr(model_generation_config, "eos_token_id", None) is not None
                else self.model.config.eos_token_id
            ),
        )
