import random
import warnings
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING

import datasets
import torch
import transformers

if TYPE_CHECKING:
    from dllm.utils.configs import DataArguments, ModelArguments, TrainingArguments


def tokenize_and_group(
    examples,
    tokenizer,
    text_field: str = "text",
    seq_length: int = 1024,
    insert_eos: bool = False,
    drop_tail: bool = True,
    add_special_tokens: bool = False,
):
    """
    Tokenize text examples and group into fixed-length sequences.

    Concatenates all tokenized text and splits into chunks of seq_length.
    Optionally drops incomplete trailing chunks.

    Args:
        examples: Batch of examples with text field.
        tokenizer: Tokenizer to use.
        text_field: Name of the text field in examples.
        seq_length: Target sequence length for chunks.
        insert_eos: If True, append EOS token to each text sample.
        drop_tail: If True, drop incomplete final chunk; if False, keep it.
        add_special_tokens: Whether to add special tokens during tokenization.

    Returns:
        Dictionary with input_ids and labels as lists of token sequences.
    """
    # 1) Tokenize (batched input)
    tokenized = tokenizer(examples[text_field], add_special_tokens=add_special_tokens)
    ids = tokenized["input_ids"]

    # --- optionally append EOS to each sample ---
    if insert_eos:
        eos_id = getattr(tokenizer, "eos_token_id")
        assert eos_id
        # append EOS only if the sample doesn't already end with it
        ids = [seq + ([] if (seq and seq[-1] == eos_id) else [eos_id]) for seq in ids]
    # ----------------------------------------------------------------

    # 2) Flatten and concatenate all token lists
    concatenated = list(chain.from_iterable(ids))
    if not concatenated:
        return {"input_ids": [], "labels": []}  # Safe return for empty batch

    # 3) Calculate the total length based on drop_tail
    if drop_tail:
        total_len = (len(concatenated) // seq_length) * seq_length
        concatenated = concatenated[:total_len]  # Truncate the last incomplete chunk
    else:
        total_len = len(concatenated)

    # Split into fixed-length chunks
    chunks = [concatenated[i : i + seq_length] for i in range(0, total_len, seq_length)]

    return {
        "input_ids": chunks,
        "labels": [c[:] for c in chunks],  # Labels are the same as input_ids
    }


def clip_row(row: dict, max_length: int, truncation: str = "right") -> dict:
    for key in ("input_ids", "labels", "attention_mask"):
        if key in row:
            if truncation == "right":
                row[key] = row[key][:max_length]
            elif truncation == "left":
                row[key] = row[key][-max_length:]
            else:
                raise NotImplementedError
    return row


def post_process_dataset(
    dataset: datasets.DatasetDict, data_args: "DataArguments"
) -> datasets.DatasetDict:
    """
    Post-process dataset by filtering or truncating sequences.

    Args:
        dataset: Dataset dictionary to process.
        data_args: Data arguments with max_length and truncation settings.

    Returns:
        Processed dataset dictionary.
    """
    if data_args.truncation == "filter":
        return dataset.filter(
            lambda row: len(row["input_ids"]) <= data_args.max_length,
            num_proc=data_args.num_proc,
            desc=f"Filtering samples with length <= {data_args.max_length}",
        )
    elif data_args.truncation == "right":
        # do this only if dataset has "prompt_len"
        if "prompt_len" in dataset.column_names["train"]:
            dataset = dataset.filter(
                lambda row: row["prompt_len"] <= data_args.max_length,
                num_proc=data_args.num_proc,
                desc=f"Filtering samples with `prompt_len` <= {data_args.max_length}",
            )
        return dataset.map(
            lambda row: clip_row(row, data_args.max_length, truncation="right"),
            num_proc=data_args.num_proc,
            desc=f"Right-truncating samples to max_length={data_args.max_length}",
        )
    else:
        raise NotImplementedError


def clip_row_streaming(row: dict, max_length: int, truncation: str = "right") -> dict:
    """Clip whole sequence OR (if prompt_len present) preserve prompt and clip only the response."""
    if truncation not in {"right", "left"}:
        raise NotImplementedError(f"Unknown truncation: {truncation}")

    def clip(seq):
        return seq[:max_length] if truncation == "right" else seq[-max_length:]

    def clip_preserve_prompt(seq, prompt_len: int):
        prompt = seq[:prompt_len]
        resp = seq[prompt_len:]
        budget = max(0, max_length - len(prompt))
        resp = resp[:budget] if truncation == "right" else resp[-budget:]
        return prompt + resp

    prompt_len = row.get("prompt_len", None)
    for k in ("input_ids", "labels", "attention_mask"):
        if k in row and isinstance(row[k], list):
            row[k] = (
                clip_preserve_prompt(row[k], prompt_len)
                if isinstance(prompt_len, int) and prompt_len >= 0
                else clip(row[k])
            )
    return row


def post_process_dataset_streaming(
    dataset: datasets.IterableDatasetDict,
    data_args: "DataArguments",
) -> datasets.IterableDatasetDict:
    """
    Post-process streaming dataset by filtering or truncating sequences.

    Similar to post_process_dataset but for streaming datasets.

    Args:
        dataset: Streaming dataset dictionary to process.
        data_args: Data arguments with max_length and truncation settings.

    Returns:
        Processed streaming dataset dictionary.
    """

    def _train_has_prompt_len_streaming(dataset: datasets.IterableDatasetDict) -> bool:
        """Replicates: 'if "prompt_len" in dataset.column_names["train"]' for streaming."""
        it = dataset["train"].take(1)
        try:
            ex = next(iter(it))
        except StopIteration:
            return False
        return "prompt_len" in ex

    mode = data_args.truncation
    max_len = data_args.max_length

    if mode == "filter":
        # Keep rows with len(input_ids) <= max_len (emulate .filter with generator map)
        def keep_if_short(row):
            if (
                "input_ids" in row
                and isinstance(row["input_ids"], list)
                and len(row["input_ids"]) <= max_len
            ):
                yield row  # keep
            # else: drop (yield nothing)

        return datasets.IterableDatasetDict(
            {name: ds.map(keep_if_short) for name, ds in dataset.items()}
        )

    elif mode == "right":
        ds_out = dataset

        # Do this only if TRAIN split has "prompt_len" (same condition as your non-streaming code)
        if _train_has_prompt_len_streaming(ds_out):

            def keep_if_prompt_fits(row):
                pl = row.get("prompt_len", None)
                if isinstance(pl, int) and pl <= max_len:
                    yield row  # keep
                elif pl is None:
                    # If a row lacks prompt_len but train had it, the non-streaming code would try to access it and fail.
                    # Here we conservatively drop such rows to mirror "requires prompt_len <= max_len".
                    return
                # else: drop

            ds_out = datasets.IterableDatasetDict(
                {name: ds.map(keep_if_prompt_fits) for name, ds in ds_out.items()}
            )

        # Then clip right (same clipping as clip_row)
        def clip_right(row):
            return clip_row(row, max_len, truncation="right")

        return datasets.IterableDatasetDict(
            {name: ds.map(clip_right) for name, ds in ds_out.items()}
        )

    else:
        raise NotImplementedError


def default_sft_map_fn(row, *, tokenizer, mask_prompt_loss: bool = True) -> dict:
    """
    Build input_ids and labels for SFT.

    Args:
        row: a dataset row with `messages`
        tokenizer: a HF tokenizer
        mask_prompt_loss: whether to mask prompt tokens (set their labels to -100)

    Returns:
        dict with keys: input_ids, labels, and optionally prompt_len
    """
    prompt_response_tokens = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False
    )
    labels = prompt_response_tokens.copy()

    if mask_prompt_loss:
        prompt_tokens = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        labels[: len(prompt_tokens)] = [-100] * len(prompt_tokens)
        return {
            "input_ids": prompt_response_tokens,
            "labels": labels,
            "prompt_len": len(prompt_tokens),
        }

    return {"input_ids": prompt_response_tokens, "labels": labels}


def prepend_bos(
    batch: dict,
    bos_token_id: int,
    label_pad_token_id: int = -100,
):
    """
    Prepend BOS to batch['input_ids'], and prepend the corresponding
    padding/ones to batch['labels'] and batch['attention_mask'] if present.
    """
    assert bos_token_id is not None, "bos_token_id must be provided"

    input_ids = batch.get("input_ids")
    bsz, _ = input_ids.shape

    # ---- input_ids ----
    bos = torch.full(
        (bsz, 1),
        bos_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    batch["input_ids"] = torch.cat([bos, input_ids], dim=1)

    # ---- labels ----
    labels = batch.get("labels")
    if labels is not None:
        ignore_labels = torch.full(
            (bsz, 1),
            label_pad_token_id,
            dtype=labels.dtype,
            device=labels.device,
        )
        batch["labels"] = torch.cat([ignore_labels, labels], dim=1)

    # ---- attention_mask ----
    attn = batch.get("attention_mask")
    if attn is not None:
        bos_attention = torch.ones(
            (bsz, 1),
            dtype=attn.dtype,
            device=attn.device,
        )
        batch["attention_mask"] = torch.cat([bos_attention, attn], dim=1)

    return batch
