from datasets import DatasetDict, load_dataset


def _build_alpaca_prompt(instruction: str, input_text: str | None) -> str:
    """
    Construct a clean text prompt from Alpaca fields.

    We intentionally *do not* include Anthropic-style role tags (e.g., "Human:", "Assistant:")
    in the returned prompt, to mirror the return shape of `load_hh_rlhf_dataset` which removes
    those tags from the prompt it returns.
    """
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()

    if input_text:
        # Keep instruction and input separated by a blank line for readability.
        return f"{instruction}\n\n{input_text}"
    else:
        return instruction


def load_dataset_alpaca(dataset_name_or_path: str) -> DatasetDict:
    """
    Load the Alpaca dataset (tatsu-lab/alpaca) and expose unified fields.

    Returns a `DatasetDict` where each split contains:
      - prompt:   Combined instruction (+ optional input), with clean formatting
      - response: The target output (model answer)

    Parameters
    ----------
    dataset_name_or_path : str
        Usually "tatsu-lab/alpaca" or a local path.
    """
    dataset = load_dataset(dataset_name_or_path)

    def map_fn(example):
        prompt = _build_alpaca_prompt(
            example.get("instruction", ""), example.get("input", "")
        )
        response = (example.get("output", "") or "").strip()
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        }

    dataset = dataset.map(
        map_fn, remove_columns=dataset["train"].column_names, num_proc=4
    )
    # make train test split
    dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
    return dataset


if __name__ == "__main__":
    from dllm.utils import resolve_with_base_env

    dataset_name_or_path = resolve_with_base_env(
        "tatsu-lab/alpaca", "BASE_DATASETS_DIR"
    )
    dataset = load_dataset_alpaca(dataset_name_or_path)
    breakpoint()
