"""
Dream pipeline namespace with lazy submodule loading.

Run import check:
    python -c "import dllm.pipelines.dream as dream; print(dream.__all__)"
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "DreamConfig",
    "DreamModel",
    "DreamTokenizer",
    "DreamSampler",
    "DreamSamplerConfig",
    "DreamTrainer",
    "utils",
]


def __getattr__(name: str):
    if name == "utils":
        module = import_module(f"{__name__}.utils")
        globals()[name] = module
        return module
    if name == "DreamConfig":
        module = import_module(f"{__name__}.models.configuration_dream")
        value = module.DreamConfig
    elif name == "DreamModel":
        module = import_module(f"{__name__}.models.modeling_dream")
        value = module.DreamModel
    elif name == "DreamTokenizer":
        module = import_module(f"{__name__}.models.tokenization_dream")
        value = module.DreamTokenizer
    elif name in ("DreamSampler", "DreamSamplerConfig"):
        module = import_module(f"{__name__}.sampler")
        value = getattr(module, name)
    elif name == "DreamTrainer":
        module = import_module(f"{__name__}.trainer")
        value = module.DreamTrainer
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
