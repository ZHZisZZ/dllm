"""Info-Gain Dream pipeline."""

from .models import InfoGainDreamConfig, InfoGainDreamModel
from .sampler import InfoGainDreamSampler, InfoGainDreamSamplerConfig

try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("infogain_dream", InfoGainDreamConfig)
    AutoModel.register(InfoGainDreamConfig, InfoGainDreamModel)
    AutoModelForMaskedLM.register(InfoGainDreamConfig, InfoGainDreamModel)
except ImportError:
    pass

__all__ = [
    "InfoGainDreamConfig",
    "InfoGainDreamModel",
    "InfoGainDreamSampler",
    "InfoGainDreamSamplerConfig",
]
