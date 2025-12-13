import accelerate
import torch
import transformers
from peft import prepare_model_for_kbit_training

from dllm.utils.configs import ModelArguments, TrainingArguments
from dllm.utils.utils import disable_caching_allocator_warmup, load_peft, print_main


def get_model(
    model_args,
    config: transformers.PretrainedConfig | None = None,
) -> transformers.PreTrainedModel:
    """
    Load a model with flexible input sources.

    Args:
        model_args: An optional dataclass or namespace containing model parameters.
        model_name_or_path: Optional direct model path or name (overrides model_args.model_name_or_path).
        dtype: Dtype (string or torch.dtype).
        load_in_4bit: Whether to load using 4-bit quantization (can override model_args.load_in_4bit).

    Returns:
        transformers.PreTrainedModel
    """
    model_name_or_path = getattr(model_args, "model_name_or_path")
    dtype = getattr(model_args, "dtype", "bfloat16")
    load_in_4bit = getattr(model_args, "load_in_4bit", False)
    attn_implementation = getattr(model_args, "attn_implementation", None)

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


def get_tokenizer(model_args) -> transformers.PreTrainedTokenizer:
    """
    Load a tokenizer with flexible input sources.

    Args:
        model_args: Optional dataclass or namespace containing model parameters.
        model: Optional model instance to configure tokenizer behavior.
        model_name_or_path: Optional direct model name or path (overrides model_args.model_name_or_path).

    Returns:
        transformers.PreTrainedTokenizer
    """
    # Lazy imports to avoid circular dependencies
    from transformers import (
        BertPreTrainedModel,
        ModernBertPreTrainedModel,
        RobertaPreTrainedModel,
    )

    from dllm.pipelines.a2d import (
        A2DLlamaLMHeadModel,
        A2DQwen2LMHeadModel,
        A2DQwen3LMHeadModel,
    )
    from dllm.pipelines.dream.models.modeling_dream import DreamModel
    from dllm.pipelines.llada2.models.modeling_llada2_moe import LLaDA2MoeModelLM
    from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
    from dllm.pipelines.llada.models.modeling_lladamoe import LLaDAMoEModelLM

    model_name_or_path = getattr(model_args, "model_name_or_path")

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
    elif issubclass(model_cls, LLaDAMoEModelLM):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|role_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif issubclass(model_cls, LLaDA2MoeModelLM):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|role_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif issubclass(model_cls, DreamModel):
        tokenizer.eot_token = "<|im_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif issubclass(
        model_cls,
        (BertPreTrainedModel, RobertaPreTrainedModel, ModernBertPreTrainedModel),
    ):
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
    elif issubclass(model_cls, A2DLlamaLMHeadModel):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|eot_id|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif issubclass(model_cls, A2DQwen2LMHeadModel):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|im_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif issubclass(model_cls, A2DQwen3LMHeadModel):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|im_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
        tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\n' }}\n    {{- '<think>\n\n</think>\n\n' }}\n{%- endif %}"
    else:
        print_main("no tokenizer customization for model class:", model_cls)
    return tokenizer
