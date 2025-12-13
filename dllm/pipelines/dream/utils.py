from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import transformers

# def top_p_logits(logits, top_p=None):
#     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
#     sorted_indices_to_remove = cumulative_probs > top_p
#     # Shift the indices to the right to keep the first token above the threshold
#     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#     sorted_indices_to_remove[..., 0] = 0

#     mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
#     mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
#     logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
#     return logits


# def top_k_logits(logits, top_k=None):
#     top_k = min(top_k, logits.size(-1))  # Safety check
#     # Remove all tokens with a probability less than the last token of the top-k
#     indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
#     logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
#     return logits


@dataclass
class DreamSFTCollator(transformers.DataCollatorForSeq2Seq):
    """
    Randomly crop response length to reduce length bias during generation.

    Reference: https://github.com/DreamLM/Dream/blob/main/src/trainer/fsdp_sft_trainer.py
    """

    perbatch_cutoff: bool = True  # Use prebatch truncation if True
    resp_cutoff_ratio: float = 0.0  # Prob. of post-collation truncation

    # -------------------------------------------------------------------------
    # 1) Pre-collation truncation (per-sample)
    # -------------------------------------------------------------------------
    def apply_perbatch_cutoff(self, features):
        """
        Randomly pick a response length from batch (`kept_len`) and trim other responses.
        Before:
        [<--promptA----><------responseA------>]
        [<--promptB-><---responseB--->]
        [<---promptC----><--respC-->]
        After:
        [<--promptA----><---respA--->]
        [<--promptB-><--respB-->]
        [<---promptC----><--respC-->]
        kept_len = 10  → trim each response to ≤10 tokens (before padding)
        """
        resp_lens = torch.tensor(
            [len(f["input_ids"]) - f["prompt_len"] for f in features], dtype=torch.long
        )
        kept_len = int(np.random.choice(resp_lens))
        for f, r_len in zip(features, resp_lens):
            remove_len = max(r_len - kept_len, 0)
            if remove_len > 0:
                # f["input_ids"] = f["input_ids"][:-remove_len]
                # f["attention_mask"] = f["attention_mask"][:-remove_len]
                # f["labels"] = f["labels"][:-remove_len]
                for key in ["input_ids", "labels", "attention_mask"]:
                    if key in f:
                        f[key] = f[key][:-remove_len]
        return features

    # -------------------------------------------------------------------------
    # 2) Post-collation truncation
    # -------------------------------------------------------------------------
    def apply_resp_cutoff(self, batch, features):
        """
        Uniformly chop tail *after padding*. All sequences truncated to new_seq_len.
        Before:
        [<--promptA----><-----respA----->]  40
        [<--promptB-><respB><----pad---->]  40
        [<---promptC----><--respC--><pad>]  40
        cutoff_len = 5
        After:
        [<--promptA----><--respA--->]   35
        [<--promptB-><respB><--pad->]   35
        [<---promptC----><--respC-->]   35
        """
        orig_seq_lens = [len(f["input_ids"]) for f in features]
        resp_lens = torch.tensor(
            [len(f["input_ids"]) - f["prompt_len"] for f in features], dtype=torch.long
        )
        min_resp_len = resp_lens.min().item()
        if min_resp_len <= 1:
            return batch

        cutoff_len = int(np.random.randint(1, min_resp_len))
        new_seq_len = max(orig_seq_lens) - cutoff_len

        for key in ["input_ids", "labels", "attention_mask"]:
            if key in batch:
                batch[key] = batch[key][:, :new_seq_len].contiguous()
        return batch

    # -------------------------------------------------------------------------
    # 3) Main call: pick truncation mode
    # -------------------------------------------------------------------------
    def __call__(self, features, return_tensors=None):
        # optional pre-collation truncation
        if self.perbatch_cutoff:
            features = self.apply_perbatch_cutoff(features)

        # always collate only the needed fields
        base = [
            {k: f[k] for k in ("input_ids", "labels", "attention_mask") if k in f}
            for f in features
        ]
        batch = super().__call__(base, return_tensors=return_tensors)

        # optional post-collation truncation
        if (
            not self.perbatch_cutoff
            and self.resp_cutoff_ratio > 0
            and np.random.rand() < self.resp_cutoff_ratio
        ):
            batch = self.apply_resp_cutoff(batch, features)

        batch.pop("prompt_len", None)
        return batch
