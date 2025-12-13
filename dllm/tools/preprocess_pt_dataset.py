"""
python dllm/tools/preprocess_pt_dataset.py
"""

import os
from dataclasses import asdict, dataclass
from functools import partial
from pprint import pprint

import datasets
import transformers
import tyro

import dllm


@dataclass
class ScriptArguments:
    """Preprocess PT dataset."""

    model_name_or_path: str = "answerdotai/ModernBERT-large"
    dataset_args: str = "OpenCoder-LLM/opc-annealing-corpus[lang:python]"  # required
    output_dir: str = "data/pt/modernbert/opc-annealing-corpus[lang:python]"  # required
    text_field: str = "text"
    max_length: int = 1024
    insert_eos: bool = True
    drop_tail: bool = True
    remove_columns: bool = False
    num_proc: int = 32

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


def preprocess_pt_dataset(
    dataset: datasets.DatasetDict,
    tokenizer: transformers.PreTrainedTokenizer,
    output_dir: str,
    text_field: str = "text",
    max_length: int = 1024,
    insert_eos: bool = True,
    drop_tail: bool = True,
    remove_columns: bool = False,
    num_proc: int = 32,
):
    processed = dataset.map(
        partial(
            dllm.utils.tokenize_and_group,
            tokenizer=tokenizer,
            text_field=text_field,
            seq_length=max_length,
            insert_eos=insert_eos,
            drop_tail=drop_tail,
        ),
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset["train"].column_names,
    )

    # Keep only the three required columns to save space.
    if remove_columns:
        keep = {"input_ids", "labels"}

        def strip_cols(ds: datasets.Dataset) -> datasets.Dataset:
            drop = [c for c in ds.column_names if c not in keep]
            return ds.remove_columns(drop) if drop else ds

        if isinstance(processed, datasets.DatasetDict):
            for split in list(processed.keys()):
                processed[split] = strip_cols(processed[split])
        else:
            processed = strip_cols(processed)

    output_dir = os.path.join(
        output_dir,
        f"max_length-{max_length}-insert_eos-{insert_eos}-drop_tail-{drop_tail}",
    )
    os.makedirs(output_dir, exist_ok=True)
    processed.save_to_disk(output_dir)
    print(f"[OK] Saved to: {output_dir}")


def main():
    # Parse with tyro
    args = tyro.cli(ScriptArguments)
    dllm.utils.print_args(args)

    tokenizer = dllm.utils.get_tokenizer(args)

    # Load your raw dataset (must contain a "messages" field per example).
    dataset = dllm.data.load_pt_dataset(args.dataset_args, streaming=False)

    preprocess_pt_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        text_field=args.text_field,
        max_length=args.max_length,
        insert_eos=args.insert_eos,
        drop_tail=args.drop_tail,
        remove_columns=args.remove_columns,
        num_proc=args.num_proc,
    )


if __name__ == "__main__":
    main()
