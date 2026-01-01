import torch
import torchmetrics


class NLLMetric(torchmetrics.aggregation.MeanMetric):
    pass


class PerplexityMetric(torchmetrics.aggregation.MeanMetric):
    def compute(self) -> torch.Tensor:
        return torch.exp(self.mean_value / self.weight)
