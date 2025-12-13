from .configuration_llada2_moe import LLaDA2MoeConfig
from .modeling_llada2_moe import LLaDA2MoeModelLM

# Register with HuggingFace Auto classes for local usage
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("llada2_moe", LLaDA2MoeConfig)
    AutoModel.register(LLaDA2MoeConfig, LLaDA2MoeModelLM)
    AutoModelForMaskedLM.register(LLaDA2MoeConfig, LLaDA2MoeModelLM)
except ImportError:
    # transformers not available or Auto classes not imported
    pass
