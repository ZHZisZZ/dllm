"""
Info-Gain Dream pipeline.

Run import check:
    python -c "from dllm.pipelines.infogain.dream import InfoGainDreamSampler"
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "InfoGainDreamConfig",
    "InfoGainDreamModel",
    "InfoGainDreamSampler",
    "InfoGainDreamSamplerConfig",
]


def _register_model_classes(config_cls, model_cls) -> None:
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

        AutoConfig.register("infogain_dream", config_cls)
        AutoModel.register(config_cls, model_cls)
        AutoModelForMaskedLM.register(config_cls, model_cls)
    except (ImportError, ValueError):
        pass


def _load_models():
    module = import_module(f"{__name__}.models")
    config_cls = module.InfoGainDreamConfig
    model_cls = module.InfoGainDreamModel
    _register_model_classes(config_cls, model_cls)
    globals()["InfoGainDreamConfig"] = config_cls
    globals()["InfoGainDreamModel"] = model_cls
    return module


def _load_sampler():
    module = import_module(f"{__name__}.sampler")
    globals()["InfoGainDreamSampler"] = module.InfoGainDreamSampler
    globals()["InfoGainDreamSamplerConfig"] = module.InfoGainDreamSamplerConfig
    return module


def __getattr__(name: str):
    if name in ("InfoGainDreamConfig", "InfoGainDreamModel"):
        _load_models()
        return globals()[name]
    if name in ("InfoGainDreamSampler", "InfoGainDreamSamplerConfig"):
        _load_sampler()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
