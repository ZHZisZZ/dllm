from typing import Optional
from contextlib import contextmanager

import torch
from torch import nn

import transformers
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.masking_utils import create_causal_mask

if transformers.utils.is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as flex_default_block_size
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
else:
    # Register a fake type to avoid crashing for annotations and `isinstance` checks
    BlockMask = torch.Tensor


class AnDLlamaConfig(transformers.LlamaConfig):
    model_type = "andi-llama"  # <- NEW model_type


class AnDLlamaModel(transformers.LlamaModel):

    def __init__(self, config: AnDLlamaConfig):
        super().__init__(config)
        self.mode = "diff" # "ar"

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if self.mode == "ar":
            # -------------------------------------------------------------
            # ORIGINAL CODE (causal mask)
            # -------------------------------------------------------------
            attention_mask = create_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            # -------------------------------------------------------------
            # ORIGINAL CODE (causal mask)
            # -------------------------------------------------------------
        elif self.mode == "diff":
            # -------------------------------------------------------------
            # NEW CODE (bidirectional, padding-only mask)
            # -------------------------------------------------------------
            # 1) If no mask is provided → treat all tokens as valid (no padding)
            if attention_mask is None:
                # No mask provided → everything valid
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2],
                    device=inputs_embeds.device,
                    dtype=torch.long,
                )

            # 2) If mask is not already a 4D attention mask → convert it
            if not (
                isinstance(attention_mask, BlockMask)
                or (isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4)
            ):
                attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)
            # -------------------------------------------------------------
            # NEW CODE (bidirectional, padding-only mask)
            # -------------------------------------------------------------
        else:
            raise NotImplementedError

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class AnDLlamaLMHeadModel(transformers.LlamaForCausalLM):
    config: AnDLlamaConfig

    def __init__(self, config):
        transformers.LlamaPreTrainedModel.__init__(self, config)
        self.model = AnDLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def mode(self) -> str:
        return self.model.mode
    
    @mode.setter
    def mode(self, v: str):
        if v not in ("ar", "diff"):
            raise NotImplementedError
        self.model.mode = v

    @contextmanager
    def mode_ctx(self, value: str):
        old = self.mode
        self.mode = value
        try:
            yield self
        finally:
            self.mode = old


transformers.AutoConfig.register("andi-llama", AnDLlamaConfig)
transformers.AutoModel.register(AnDLlamaConfig, AnDLlamaLMHeadModel)
transformers.AutoModelForMaskedLM.register(AnDLlamaConfig, AnDLlamaLMHeadModel)


if __name__ == "__main__":
    import dllm
    import torch
    from transformers import AutoModel

    # Load a config from a local path (either a directory containing config.json, or the file itself)
    config_path = dllm.utils.resolve_with_base_env(
        "meta-llama/Meta-Llama-3-8B", "BASE_MODELS_DIR"
    )
    config = AnDLlamaConfig.from_pretrained(config_path)
    if hasattr(config, "auto_map"):
        delattr(config, "auto_map")
    if hasattr(config, "architectures"):
        delattr(config, "architectures")

    torch.set_default_device("cuda")
    model = AnDLlamaModel(config)
    model.save_pretrained("models-tmp/andi-llama")
    auto_model = AutoModel.from_pretrained("models-tmp/andi-llama")
