"""
Pipeline namespace with lazy submodule loading.

This keeps `dllm.pipelines.fastdllm` / `dllm.pipelines.infogain` style access
working without importing every optional pipeline dependency at package import
time.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "a2d",
    "bert",
    "dream",
    "editflow",
    "fastdllm",
    "infogain",
    "llada",
    "llada2",
    "llada21",
    "rl",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
