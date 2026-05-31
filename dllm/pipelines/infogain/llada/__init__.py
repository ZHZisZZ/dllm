"""
Info-Gain LLaDA pipeline.

Run import check:
    python -c "from dllm.pipelines.infogain.llada import InfoGainLLaDASampler"
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "InfoGainLLaDAConfig",
    "InfoGainLLaDAModelLM",
    "InfoGainLLaDASampler",
    "InfoGainLLaDASamplerConfig",
]


def _register_model_classes(config_cls, model_cls) -> None:
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

        AutoConfig.register("infogain_llada", config_cls)
        AutoModel.register(config_cls, model_cls)
        AutoModelForMaskedLM.register(config_cls, model_cls)
    except (ImportError, ValueError):
        pass


def _load_models():
    module = import_module(f"{__name__}.models")
    config_cls = module.InfoGainLLaDAConfig
    model_cls = module.InfoGainLLaDAModelLM
    _register_model_classes(config_cls, model_cls)
    globals()["InfoGainLLaDAConfig"] = config_cls
    globals()["InfoGainLLaDAModelLM"] = model_cls
    return module


def _load_sampler():
    module = import_module(f"{__name__}.sampler")
    globals()["InfoGainLLaDASampler"] = module.InfoGainLLaDASampler
    globals()["InfoGainLLaDASamplerConfig"] = module.InfoGainLLaDASamplerConfig
    return module


def __getattr__(name: str):
    if name in ("InfoGainLLaDAConfig", "InfoGainLLaDAModelLM"):
        _load_models()
        return globals()[name]
    if name in ("InfoGainLLaDASampler", "InfoGainLLaDASamplerConfig"):
        _load_sampler()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
