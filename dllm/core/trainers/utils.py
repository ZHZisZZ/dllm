import math

import torch
import transformers


class EpochPPLMeter(transformers.TrainerCallback):
    """
    Keeps running sums for dataset-level NLL/token and logs PPL once per epoch.

    Usage:
      - Trainer calls: self.ppl_meter.update(split, nll_sum, token_cnt)
      - Callback hooks:
          * on_epoch_begin: reset train accumulators
          * on_epoch_end: finalize+log train PPL
          * on_evaluate:   finalize+log eval  PPL (one per evaluate call)
    """

    def __init__(
        self,
        trainer: "transformers.Trainer",
        train_prefix: str = "train",
        eval_prefix: str = "eval",
    ):
        self.trainer = trainer
        self.train_prefix = train_prefix
        self.eval_prefix = eval_prefix

        self._train_nll_sum = 0.0
        self._train_token_cnt = 0.0
        self._eval_nll_sum = 0.0
        self._eval_token_cnt = 0.0

    def reset(self, split: str) -> None:
        if split == "train":
            self._train_nll_sum = 0.0
            self._train_token_cnt = 0.0
        elif split == "eval":
            self._eval_nll_sum = 0.0
            self._eval_token_cnt = 0.0
        else:
            raise ValueError(f"Unknown split={split}")

    def update(self, split: str, nll_sum: torch.Tensor, token_cnt: torch.Tensor) -> None:
        # detach -> float64 -> python float
        nll_sum_f = float(nll_sum.detach().double().cpu().item())
        tok_cnt_f = float(token_cnt.detach().double().cpu().item())

        if split == "train":
            self._train_nll_sum += nll_sum_f
            self._train_token_cnt += tok_cnt_f
        elif split == "eval":
            self._eval_nll_sum += nll_sum_f
            self._eval_token_cnt += tok_cnt_f
        else:
            raise ValueError(f"Unknown split={split}")

    def _finalize(self, split: str):
        """
        All-reduce (sum) across processes, then compute:
            mean_nll = total_nll / total_tokens
            ppl      = exp(mean_nll)

        Returns (mean_nll, ppl) as python floats, or (None, None) if no tokens.
        Also resets that split after finalizing.
        """
        if split == "train":
            local_nll, local_tok = self._train_nll_sum, self._train_token_cnt
            self.reset("train")
        elif split == "eval":
            local_nll, local_tok = self._eval_nll_sum, self._eval_token_cnt
            self.reset("eval")
        else:
            raise ValueError(f"Unknown split={split}")

        if local_tok <= 0.0:
            return None, None

        device = getattr(self.trainer.args, "device", torch.device("cpu"))
        stats = torch.tensor([local_nll, local_tok], device=device, dtype=torch.float64)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

        total_nll = float(stats[0].item())
        total_tok = float(stats[1].item())
        if total_tok <= 0.0:
            return None, None

        mean_nll = total_nll / total_tok
        ppl = math.exp(mean_nll)
        return mean_nll, ppl

    # ---- callback hooks ----

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.reset("train")
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        mean_nll, ppl = self._finalize("train")
        if mean_nll is not None and self.trainer.is_world_process_zero():
            logs = {f"{self.train_prefix}_nll": mean_nll, f"{self.train_prefix}_ppl": ppl}
            self.trainer.log(logs)
            print(f"[epoch {state.epoch}] {self.train_prefix}_nll={mean_nll:.6f} {self.train_prefix}_ppl={ppl:.6f}")
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        mean_nll, ppl = self._finalize("eval")
        if mean_nll is not None and self.trainer.is_world_process_zero():
            logs = {f"{self.eval_prefix}_nll": mean_nll, f"{self.eval_prefix}_ppl": ppl}
            self.trainer.log(logs)
            print(f"[epoch {state.epoch}] {self.eval_prefix}_nll={mean_nll:.6f} {self.eval_prefix}_ppl={ppl:.6f}")
        return control
