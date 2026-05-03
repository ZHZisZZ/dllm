"""
Info-Gain masked diffusion sampling (LLaDA + Dream).

Upstream reference: https://github.com/yks23/Information-Gain-Sampler
"""

from . import dream, llada

__all__ = ["dream", "llada"]
