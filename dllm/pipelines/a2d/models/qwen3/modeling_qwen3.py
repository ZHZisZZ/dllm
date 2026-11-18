"""Qwen3 with A2D attention masks.

From the repo root, verify BD3LM masking with:
    python -m pytest scripts/tests/attention/test_bd3lm.py -k boolean_mask_blocks_attention
"""

from typing import Optional

import torch
from torch import nn

import transformers
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

if transformers.utils.is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as flex_default_block_size
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
else:
    # Register a fake type to avoid crashing for annotations and `isinstance` checks.
    # Must NOT be torch.Tensor — that would make isinstance(any_tensor, BlockMask) always
    # True and silently break the mask-conversion logic below.
    class BlockMask:
        pass

class A2DQwen3Config(transformers.Qwen3Config):
    model_type = "a2d-qwen3"  # <- NEW model_type


class A2DQwen3Model(transformers.Qwen3Model):

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        """
        # -------------------------------------------------------------
        # ORIGINAL CODE (causal mask)
        # -------------------------------------------------------------
        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        # -------------------------------------------------------------
        # ORIGINAL CODE (causal mask)
        # -------------------------------------------------------------
        """
        # -------------------------------------------------------------
        # NEW CODE (bidirectional, padding-only mask)
        # -------------------------------------------------------------
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # 1) If no mask is provided → treat all tokens as valid (no padding)
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2],
                    device=inputs_embeds.device,
                    dtype=torch.long
                )

            # 2) If mask is not already a 4D attention mask → convert it
            if not (
                isinstance(attention_mask, BlockMask)
                or (isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4)
            ):
                attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)
            elif (
                self.config._attn_implementation == "eager"
                and isinstance(attention_mask, torch.Tensor)
                and attention_mask.dtype == torch.bool
            ):
                # Eager adds the mask to attention scores. A finite bias keeps
                # softmax defined for fully masked padding queries.
                dtype = inputs_embeds.dtype
                attention_mask = torch.zeros_like(attention_mask, dtype=dtype).masked_fill_(
                    ~attention_mask, torch.finfo(dtype).min
                )

            # 3) Build causal mask mapping used by the attention layers
            causal_mask_mapping = {"full_attention": attention_mask}

            # Sliding-window layers share the same non-causal mask
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = attention_mask
        # -------------------------------------------------------------
        # NEW CODE (bidirectional, padding-only mask)
        # -------------------------------------------------------------

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class A2DQwen3LMHeadModel(transformers.Qwen3ForCausalLM):
    config: A2DQwen3Config

    def __init__(self, config):
        transformers.Qwen3PreTrainedModel.__init__(self, config)
        self.model = A2DQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


transformers.AutoConfig.register("a2d-qwen3", A2DQwen3Config)
transformers.AutoModel.register(A2DQwen3Config, A2DQwen3LMHeadModel)
transformers.AutoModelForMaskedLM.register(A2DQwen3Config, A2DQwen3LMHeadModel)


if __name__ == "__main__":
    import dllm
    import torch
    from transformers import AutoModel

    # Load a config from a local path (either a directory containing config.json, or the file itself)
    config_path = dllm.utils.resolve_with_base_env(
        "Qwen/Qwen3-0.6B-Base", "BASE_MODELS_DIR"
    )
    config = A2DQwen3Config.from_pretrained(config_path)
    if hasattr(config, "auto_map"):
        delattr(config, "auto_map")
    if hasattr(config, "architectures"):
        delattr(config, "architectures")

    torch.set_default_device("cuda")
    model = A2DQwen3LMHeadModel(config)
    model.save_pretrained("models-tmp/a2d-qwen3")
    auto_model = AutoModel.from_pretrained("models-tmp/a2d-qwen3")

