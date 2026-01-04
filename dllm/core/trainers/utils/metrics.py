import torch
import torchmetrics


class NLLMetric(torchmetrics.aggregation.MeanMetric):
    pass


class PPLMetric(NLLMetric):
    def compute(self) -> torch.Tensor:
        mean_nll = super().compute()
        return torch.exp(mean_nll)
