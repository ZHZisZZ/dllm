"""Info-Gain LLaDA pipeline."""

from .models import InfoGainLLaDAConfig, InfoGainLLaDAModelLM
from .sampler import InfoGainLLaDASampler, InfoGainLLaDASamplerConfig

try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("infogain_llada", InfoGainLLaDAConfig)
    AutoModel.register(InfoGainLLaDAConfig, InfoGainLLaDAModelLM)
    AutoModelForMaskedLM.register(InfoGainLLaDAConfig, InfoGainLLaDAModelLM)
except ImportError:
    pass

__all__ = [
    "InfoGainLLaDAConfig",
    "InfoGainLLaDAModelLM",
    "InfoGainLLaDASampler",
    "InfoGainLLaDASamplerConfig",
]
