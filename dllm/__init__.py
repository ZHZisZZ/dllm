"""
Top-level dllm package with lazy submodule loading.

Run import check:
    python -c "import dllm; print(dllm.__all__)"
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["core", "data", "pipelines", "utils"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
