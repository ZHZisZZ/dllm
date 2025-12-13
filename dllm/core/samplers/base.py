from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler


@dataclass
class SamplerOutput:
    sequences: torch.Tensor
    histories: list[torch.Tensor] | None = None


@dataclass
class SamplerConfig:
    return_dict: bool = False


@dataclass
class BaseSampler(ABC):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    scheduler: BaseAlphaScheduler | None = None

    def __post_init__(self):
        if self.scheduler is None:
            self.scheduler = LinearAlphaScheduler()

    @abstractmethod
    @torch.no_grad()
    def sample(
        self,
        prompts: list[torch.Tensor, list],
        config: SamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor, list],
        config: SamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput:
        raise NotImplementedError
