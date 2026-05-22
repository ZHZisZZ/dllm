from types import SimpleNamespace

import accelerate
import torch
import transformers
from peft import prepare_model_for_kbit_training

from dllm.utils.configs import ModelArguments, TrainingArguments
from dllm.utils.utils import disable_caching_allocator_warmup, load_peft, print_main


def get_model(
    model_args: ModelArguments | None = None,
    config: transformers.PretrainedConfig | None = None,
    **kwargs,
) -> transformers.PreTrainedModel:
    """
    Load a model with flexible input sources.

    Args:
        model_args: Dataclass or namespace containing model parameters, or None to use **kwargs.
        config: Optional transformers.PretrainedConfig to use instead of loading from the checkpoint.
        **kwargs: Override or supply params when model_args is None (e.g. model_name_or_path, dtype).

    Returns:
        transformers.PreTrainedModel
    """
    model_args = model_args or ModelArguments()
    model_name_or_path = kwargs.get(
        "model_name_or_path", getattr(model_args, "model_name_or_path", None)
    )
    dtype = kwargs.get("dtype", getattr(model_args, "dtype", "bfloat16"))
    load_in_4bit = kwargs.get(
        "load_in_4bit", getattr(model_args, "load_in_4bit", False)
    )
    attn_implementation = kwargs.get(
        "attn_implementation", getattr(model_args, "attn_implementation", None)
    )

    # Device map: skip when ZeRO-3
    device_map = (
        {"": accelerate.PartialState().local_process_index}
        if not transformers.modeling_utils.is_deepspeed_zero3_enabled()
        and torch.cuda.is_available()
        else None
    )

    quant_config = None
    if load_in_4bit and transformers.utils.is_bitsandbytes_available():
        quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    params = {
        "dtype": dtype,
        "device_map": device_map,
        "quantization_config": quant_config,
        "attn_implementation": attn_implementation,
        "config": config,
    }

    try:
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, **params
        )
    except Exception:
        model = transformers.AutoModel.from_pretrained(model_name_or_path, **params)

    # --- if quantized, prepare for LoRA / QLoRA training ---
    if load_in_4bit and quant_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # Optionally train with lora
    model = load_peft(model, model_args)

    return model


def get_tokenizer(
    model_args: ModelArguments | None = None, **kwargs
) -> transformers.PreTrainedTokenizer:
    """
    Load a tokenizer with flexible input sources.

    Args:
        model_args: Namespace/dataclass containing at least model_name_or_path, or None to use **kwargs.
        **kwargs: Override or supply params when model_args is None (e.g. model_name_or_path).

    Returns:
        transformers.PreTrainedTokenizer
    """
    # Lazy imports to avoid circular dependencies
    from transformers import (
        BertPreTrainedModel,
        RobertaPreTrainedModel,
    )
    try:
        from transformers import ModernBertPreTrainedModel
    except ImportError:
        ModernBertPreTrainedModel = None

    try:
        from dllm.pipelines.a2d import (
            A2DLlamaLMHeadModel,
            A2DQwen2LMHeadModel,
            A2DQwen3LMHeadModel,
        )
    except ImportError:
        A2DLlamaLMHeadModel = None
        A2DQwen2LMHeadModel = None
        A2DQwen3LMHeadModel = None
    try:
        from dllm.pipelines.dream.models.modeling_dream import DreamModel
    except ImportError:
        DreamModel = None
    try:
        from dllm.pipelines.llada2.models import modeling_llada2_moe as llada2_moe_mod
    except ImportError:
        llada2_moe_mod = None
    try:
        from dllm.pipelines.llada21.models import modeling_llada21_moe as llada21_moe_mod
    except ImportError:
        llada21_moe_mod = None
    from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
    from dllm.pipelines.llada.models.modeling_lladamoe import LLaDAMoEModelLM

    model_args = model_args or ModelArguments()
    model_name_or_path = kwargs.get(
        "model_name_or_path", getattr(model_args, "model_name_or_path", None)
    )

    # ---------------- Tokenizer loading ----------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
    )

    assert tokenizer.eos_token is not None or tokenizer.pad_token is not None

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.eos_token:
        tokenizer.eos_token = tokenizer.pad_token
    if not tokenizer.bos_token:
        tokenizer.bos_token = tokenizer.pad_token

    # If model is not provided, return as-is
    model_cfg = transformers.AutoConfig.from_pretrained(model_name_or_path)
    model_cls = transformers.AutoModel._model_mapping[type(model_cfg)]

    bert_model_classes = [BertPreTrainedModel, RobertaPreTrainedModel]
    if ModernBertPreTrainedModel is not None:
        bert_model_classes.append(ModernBertPreTrainedModel)
    llada_moe_model_classes = [LLaDAMoEModelLM]
    if llada2_moe_mod is not None:
        llada_moe_model_classes.append(llada2_moe_mod.LLaDA2MoeModelLM)
    if llada21_moe_mod is not None:
        llada_moe_model_classes.append(llada21_moe_mod.LLaDA2MoeModelLM)

    # ---------------- Model-specific customization ----------------
    if issubclass(model_cls, LLaDAModelLM):
        tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
        tokenizer.eot_token = "<|eot_id|>"
        # tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token) # can not do this for llada base directly
        # TODO: for llada base, add special_tokens = {"<|start_header_id|>": 126346, "<|end_header_id|>": 126347, "<|eot_id|>": 126348}
        # fix bugs in chat template
        tokenizer.chat_template = """\
{% set loop_messages = messages %}
{% for message in loop_messages %}
{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] | trim }}<|eot_id|>
{%- endfor %}
{% if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}
"""
    elif issubclass(model_cls, tuple(llada_moe_model_classes)):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|role_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif DreamModel is not None and issubclass(model_cls, DreamModel):
        tokenizer.eot_token = "<|im_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif issubclass(model_cls, tuple(bert_model_classes)):
        tokenizer.eot_token = "[/Answer]"
        tokenizer.chat_template = """\
{% if messages[0]['role'] == 'system' %}
[SYS]
{{ messages[0]['content'] | trim }}
[/SYS]

{% set loop_messages = messages[1:] %}
{% else %}
{% set loop_messages = messages %}
{% endif -%}
{%- for message in loop_messages %}
{% if message['role'] == 'user' %}
[Question]
{{ message['content'] | trim }}
[/Question]

{% elif message['role'] == 'assistant' %}
[Answer]
{{ message['content'] | trim }}
[/Answer]

{% endif %}
{% endfor -%}
{%- if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
[Answer]
{% endif %}
"""
    elif A2DLlamaLMHeadModel is not None and issubclass(model_cls, A2DLlamaLMHeadModel):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|eot_id|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif any(cls is not None and issubclass(model_cls, cls) for cls in (A2DQwen2LMHeadModel, A2DQwen3LMHeadModel)):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|im_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
        # When enable_thinking is not passed, default to False so the chat template
        # appends <think></think> (add think). Only skip that when enable_thinking=True.
        _orig_apply_chat_template = tokenizer.apply_chat_template

        def _apply_chat_template(*args, **kwargs):
            if "enable_thinking" not in kwargs:
                kwargs["enable_thinking"] = False
            try:
                return _orig_apply_chat_template(*args, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                return _orig_apply_chat_template(*args, **kwargs)

        tokenizer.apply_chat_template = _apply_chat_template
    else:
        print_main("no tokenizer customization for model class:", model_cls)
    return tokenizer
