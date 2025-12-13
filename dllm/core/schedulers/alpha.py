from __future__ import annotations

import dataclasses
import math
from typing import Any, ClassVar, Union

import torch

Number = Union[float, torch.Tensor]


# ---------------- Registry-enabled Base ---------------- #
@dataclasses.dataclass
class BaseAlphaScheduler:
    """
    Base class for alpha schedulers in diffusion language models.

    Alpha schedulers define the masking rate α(t) as a function of diffusion time t ∈ [0,1].
    Subclasses are automatically registered and can be instantiated by name.

    To implement a custom scheduler, inherit from this class and implement:
    - _alpha(t): Compute α(t) for a tensor of timesteps
    - _alpha_derivative(t): Compute dα/dt for a tensor of timesteps

    Example:
        @dataclasses.dataclass
        class CustomScheduler(BaseAlphaScheduler):
            def _alpha(self, t):
                return 1 - t**2
            def _alpha_derivative(self, t):
                return -2 * t
    """

    __registry__: ClassVar[dict[str, type[BaseAlphaScheduler]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseAlphaScheduler.__registry__[cls.__name__] = cls
        BaseAlphaScheduler.__registry__[cls.__name__.lower()] = cls

    # Make instances callable (sched(i) -> alpha(i))
    def __call__(self, t: Number) -> Number:
        return self.alpha(t)

    # ---- common API ----
    def alpha(self, i: Number) -> Number:
        i_t = torch.as_tensor(
            i,
            dtype=torch.float32,
            device=i.device if isinstance(i, torch.Tensor) else None,
        )
        if not torch.all((0.0 <= i_t) & (i_t <= 1.0)):
            raise ValueError(f"i={i} not in [0,1]")
        out = self._alpha(i_t)
        return out.item() if isinstance(i, float) else out

    def alpha_derivative(self, i: Number) -> Number:
        i_t = torch.as_tensor(
            i,
            dtype=torch.float32,
            device=i.device if isinstance(i, torch.Tensor) else None,
        )
        if not torch.all((0.0 <= i_t) & (i_t <= 1.0)):
            raise ValueError(f"i={i} not in [0,1]")
        out = self._alpha_derivative(i_t)
        return out.item() if isinstance(i, float) else out

    def reverse_mask_prob(self, s: Number, t: Number) -> Number:
        t_t = torch.as_tensor(
            t,
            dtype=torch.float32,
            device=t.device if isinstance(t, torch.Tensor) else None,
        )
        s_t = torch.as_tensor(
            s,
            dtype=torch.float32,
            device=s.device if isinstance(s, torch.Tensor) else None,
        )
        if not torch.all((0.0 <= s_t) & (s_t < 1.0) & (0.0 < t_t) & (t_t <= 1.0)):
            raise ValueError(f"(t={t}, s={s}) out of range")
        if not torch.all(s_t < t_t):
            raise ValueError(f"Require s < t elementwise, but got (t={t}, s={s})")
        out = (1 - self(s_t)) / (1 - self(t_t))
        return out.item() if isinstance(t, float) and isinstance(s, float) else out

    def weight(self, i: Number) -> Number:
        # w(t) = - α'(t) / (1 - α(t))
        return -self.alpha_derivative(i) / (1 - self.alpha(i) + 1e-6)

    # ---- hooks implemented by subclasses ----
    def _alpha(self, i: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _alpha_derivative(self, i: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------- Implementations ---------------- #


@dataclasses.dataclass
class LinearAlphaScheduler(BaseAlphaScheduler):
    def _alpha(self, i: torch.Tensor) -> torch.Tensor:
        return 1 - i

    def _alpha_derivative(self, i: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(i)


@dataclasses.dataclass
class CosineAlphaScheduler(BaseAlphaScheduler):
    def _alpha(self, i: torch.Tensor) -> torch.Tensor:
        return 1 - torch.cos((math.pi / 2) * (1 - i))

    def _alpha_derivative(self, i: torch.Tensor) -> torch.Tensor:
        return -(math.pi / 2) * torch.sin((math.pi / 2) * (1 - i))


# ---------------- Factory helpers ---------------- #


def get_alpha_scheduler_class(name: str) -> type[BaseAlphaScheduler]:
    """Return the scheduler class by name (case-insensitive)."""
    cls = BaseAlphaScheduler.__registry__.get(
        name
    ) or BaseAlphaScheduler.__registry__.get(name.lower())
    if cls is None:
        available = sorted(k for k in BaseAlphaScheduler.__registry__ if k[0].isupper())
        raise ValueError(f"Unknown scheduler '{name}'. Available: {available}")
    return cls


def make_alpha_scheduler(name: str, **kwargs: Any) -> BaseAlphaScheduler:
    """Instantiate a scheduler by name with optional kwargs."""
    cls = get_alpha_scheduler_class(name)
    return cls(**kwargs)


# ---------------- Example usage ---------------- #

if __name__ == "__main__":
    lin_sched = make_alpha_scheduler("LinearalphaScheduler")
    print("Linear α(0.5):", lin_sched.alpha(0.5))
    print("Linear w(0.5):", lin_sched.weight(0.5))
    print("Linear α([.25,.5,.75]):", lin_sched.alpha(torch.tensor([0.25, 0.5, 0.75])))
    print("Linear w([.25,.5,.75]):", lin_sched.weight(torch.tensor([0.25, 0.5, 0.75])))
    print("==========================================")
    cos_sched = make_alpha_scheduler("CosinealphaScheduler")
    print("Cosine α(0.5):", cos_sched.alpha(0.5))
    print("Cosine w(0.5):", cos_sched.weight(0.5))
    print("Cosine α([.25,.5,.75]):", cos_sched.alpha(torch.tensor([0.25, 0.5, 0.75])))
    print("Cosine w([.25,.5,.75]):", cos_sched.weight(torch.tensor([0.25, 0.5, 0.75])))
