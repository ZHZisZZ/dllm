"""
Example:

PYTHONPATH=. python dllm/tools/preprocess_sft_dataset.py \
    --model_name_or_path "Dream-org/Dream-v0-Base-7B" \
    --sft_map_fn_path "dllm.utils.ar_mdlm_sft_map_fn" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "data/sft/dream/tulu-3-sft-mixture" \
    --num_proc 64
"""

import importlib
import os
from dataclasses import dataclass
from functools import partial

import datasets
import tyro

import dllm


@dataclass
class ScriptArguments:
    """Preprocess SFT dataset."""

    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"
    sft_map_fn_path: str = "dllm.utils.default_sft_map_fn"
    dataset_args: str = "HuggingFaceTB/smoltalk"  # required
    output_dir: str = "data/sft/llada/smoltalk"  # required
    mask_prompt_loss: bool = True  # Mask prompt tokens in labels with -100
    num_proc: int = 32
    remove_columns: bool = False

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


def preprocess_sft_dataset(
    dataset: datasets.DatasetDict,
    map_fn: callable,
    output_dir: str,
    remove_columns: bool = False,
    num_proc: int = 32,
):
    processed = dataset.map(
        map_fn,
        batched=False,
        num_proc=num_proc,
        load_from_cache_file=True,
        writer_batch_size=512,
        desc="offline preprocessing",
    )

    # Keep only the three required columns to save space.
    if remove_columns:
        keep = {"input_ids", "labels", "prompt_len", "attention_mask"}

        def strip_cols(ds: datasets.Dataset) -> datasets.Dataset:
            drop = [c for c in ds.column_names if c not in keep]
            return ds.remove_columns(drop) if drop else ds

        if isinstance(processed, datasets.DatasetDict):
            for split in list(processed.keys()):
                processed[split] = strip_cols(processed[split])
        else:
            processed = strip_cols(processed)

    os.makedirs(output_dir, exist_ok=True)
    processed.save_to_disk(output_dir)
    print(f"[OK] Saved to: {output_dir}")


def main():
    # Parse with tyro
    args = tyro.cli(ScriptArguments)
    dllm.utils.print_args(args)

    tokenizer = dllm.utils.get_tokenizer(args)

    # Load your raw dataset (must contain a "messages" field per example).
    dataset = dllm.data.load_sft_dataset(args.dataset_args)

    # 4. Dynamically import the function based on the argument
    try:
        # Split the path into module and function name
        module_path, function_name = args.sft_map_fn_path.rsplit(".", 1)

        # Import the module
        module = importlib.import_module(module_path)

        # Get the function from the module
        sft_map_fn = getattr(module, function_name)

    except (ImportError, AttributeError, ValueError) as e:
        print(f"Error: Could not import '{args.sft_map_fn_path}'.")
        print(f"Details: {e}")
        return

    map_fn = partial(
        sft_map_fn,
        tokenizer=tokenizer,
        mask_prompt_loss=args.mask_prompt_loss,
    )
    preprocess_sft_dataset(
        dataset=dataset,
        map_fn=map_fn,
        output_dir=args.output_dir,
        remove_columns=args.remove_columns,
        num_proc=args.num_proc,
    )


if __name__ == "__main__":
    main()
