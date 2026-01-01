from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
import copy

import torch
import transformers
import torchmetrics


def _ddp_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class BaseMetricsCallback(transformers.TrainerCallback):
    """
    Generic split-aware metric accumulator for HF Trainer.

    Fixes vs old version:
      1) Per-split metrics are independent (deep-copied) to avoid train/eval contamination.
      2) DDP-safe: sync/compute/reset run on ALL ranks; only rank0 logs/prints (no deadlock).
      3) Metrics are moved to trainer device (avoids CPU/GPU mismatch).
      4) Optional dtype set if metric supports it.

    Smart key prefixing:
      - split == "train": no prefix (e.g., "loss", "ppl")
      - otherwise       : f"{split}_" prefix (e.g., "eval_loss", "test_ppl")

    You provide:
      - metrics_map:
          * {name: Metric}            -> broadcast to all splits (deep-copied per split)
          * {split: {name: Metric}}   -> per-split metrics
    update():
      - calls metric.update(*args, **kwargs) by default
    """

    def __init__(
        self,
        trainer: "transformers.Trainer",
        splits: Iterable[str] = ("train", "eval"),
        metrics_map: Optional[Dict[str, Any]] = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.trainer = trainer
        self.splits = tuple(splits)
        metrics_map = metrics_map or {}

        # Create per-split independent metric dicts
        self._metrics: Dict[str, Dict[str, torchmetrics.Metric]] = {}

        # Detect broadcast map: {name: Metric}
        is_broadcast = len(metrics_map) > 0 and all(
            isinstance(v, torchmetrics.Metric) for v in metrics_map.values()
        )

        device = getattr(self.trainer.args, "device", torch.device("cpu"))

        for split in self.splits:
            if is_broadcast:
                # IMPORTANT: deepcopy so each split has independent state
                mdict = {k: copy.deepcopy(v) for k, v in metrics_map.items()}
            else:
                mdict = {
                    k: copy.deepcopy(v) for k, v in metrics_map.get(split, {}).items()
                }

            # Configure dtype / device
            for m in mdict.values():
                # Many torchmetrics ignore this, but keep your hook
                if hasattr(m, "set_dtype"):
                    m.set_dtype(dtype)
                # Ensure state buffers are on the right device
                try:
                    m.to(device)
                except Exception:
                    pass

            self._metrics[split] = mdict

    # ---------- key naming ----------

    @staticmethod
    def key_for(split: str, name: str) -> str:
        return name if split == "train" else f"{split}_{name}"

    # ---------- lifecycle ----------

    def reset(self, split: str) -> None:
        for m in self._metrics[split].values():
            m.reset()

    @torch.no_grad()
    def update(self, split: str, *args, **kwargs) -> None:
        for m in self._metrics[split].values():
            m.update(*args, **kwargs)

    @torch.no_grad()
    def finalize(self, split: str) -> Dict[str, float]:
        """
        DDP-safe finalize:
          - Must be called on ALL ranks (because sync uses collectives).
          - Returns local dict of python floats.
          - Resets split metrics.
        """
        mdict = self._metrics[split]

        # Make sure metrics live on current device (in case trainer device changes)
        device = getattr(self.trainer.args, "device", torch.device("cpu"))
        for m in mdict.values():
            try:
                m.to(device)
            except Exception:
                pass

        # Sync across ranks (collectives) -- MUST run on all ranks
        if _ddp_initialized():
            for m in mdict.values():
                # torchmetrics usually has sync/unsync; prefer sync if available
                if hasattr(m, "sync"):
                    m.sync()

        out: Dict[str, float] = {}
        for name, m in mdict.items():
            v = m.compute()
            if isinstance(v, torch.Tensor):
                if v.numel() == 0:
                    continue
                v = v.detach()
                v = v.item() if v.numel() == 1 else v.double().mean().cpu().item()
            out[name] = float(v)

        # IMPORTANT: reset after compute so next window starts clean
        self.reset(split)
        return out

    @torch.no_grad()
    def log_and_print(
        self,
        state: transformers.TrainerState,
        splits: Iterable[str] | None = None,
    ) -> None:
        """
        DDP-safe:
          - finalize() (and thus sync/compute/reset) runs on ALL ranks
          - only rank0 logs/prints
        """
        splits = self.splits if splits is None else tuple(splits)

        # All ranks finalize (avoid DDP deadlock)
        all_vals: Dict[str, Dict[str, float]] = {}
        for split in splits:
            if split in self._metrics:
                all_vals[split] = self.finalize(split)

        # Only rank0 logs/prints
        if not self.trainer.is_world_process_zero():
            return

        logs: Dict[str, float] = {}
        for split, vals in all_vals.items():
            logs.update({self.key_for(split, k): v for k, v in vals.items()})

        if logs:
            self.trainer.log(logs)
            print(
                f"[step {state.global_step} epoch {state.epoch}] "
                + " ".join(f"{k}={v:.6f}" for k, v in logs.items())
            )

    # ---------- HF callback hooks (optional defaults) ----------

    def on_train_begin(self, args, state, control, **kwargs):
        if "train" in self._metrics:
            self.reset("train")
        return control

    def on_evaluate_begin(self, args, state, control, **kwargs):
        if "eval" in self._metrics:
            self.reset("eval")
        return control


class OnEvaluateMetricsCallback(BaseMetricsCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Log both train + eval by default (matches your previous behavior).
        self.log_and_print(state, splits=("train", "eval"))
        return control
