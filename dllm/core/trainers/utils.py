import math
import torch
import transformers


class EpochPPLMeter(transformers.TrainerCallback):
    """
    Keeps running sums for dataset-level NLL/token and logs PPL.
    Convention:
      - Train: keys are unprefixed, e.g. "diff_nll", "diff_ppl"
      - Eval : keys are prefixed with "eval_", e.g. "eval_diff_nll", "eval_diff_ppl"
    """

    def __init__(self, trainer: "transformers.Trainer"):
        self.trainer = trainer

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

    def update(
        self, split: str, nll_sum: torch.Tensor, token_cnt: torch.Tensor
    ) -> None:
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
        Returns (mean_nll, ppl) or (None, None) if no tokens.
        Resets the split accumulators when called.
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

    def on_train_begin(self, args, state, control, **kwargs):
        self.reset("train")
        return control

    def on_evaluate_begin(self, args, state, control, **kwargs):
        self.reset("eval")
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        train_mean_nll, train_ppl = self._finalize("train")
        eval_mean_nll, eval_ppl = self._finalize("eval")

        if self.trainer.is_world_process_zero():
            logs = {}

            # TRAIN: NO "train_" prefix
            if train_mean_nll is not None:
                logs.update(
                    {
                        "diff_nll": train_mean_nll,
                        "diff_ppl": train_ppl,
                    }
                )

            # EVAL: MUST be "eval_" prefixed
            if eval_mean_nll is not None:
                logs.update(
                    {
                        "eval_diff_nll": eval_mean_nll,
                        "eval_diff_ppl": eval_ppl,
                    }
                )

            if logs:
                self.trainer.log(logs)
                print(
                    f"[step {state.global_step} epoch {state.epoch}] "
                    + " ".join(f"{k}={v:.6f}" for k, v in logs.items())
                )

        return control
