from datasets import DatasetDict, load_dataset


def _extract_first_turn(messages: list[dict[str, str]]) -> dict[str, str] | None:
    """
    Given a list of chat messages like:
      [{"role": "user", "content": "..."},
       {"role": "assistant", "content": "..."},
       ...]
    return a dict with the first user/assistant exchange as:
      {"prompt": <user content>, "response": <assistant content>}
    If no valid first turn exists, return None.
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    # Find the first user message and the first assistant *after* that user msg
    # (Most entries start as [user, assistant, ...], but we guard anyway.)
    user_idx = None
    for i, m in enumerate(messages):
        if (
            isinstance(m, dict)
            and m.get("role") == "user"
            and isinstance(m.get("content"), str)
        ):
            user_idx = i
            break
    if user_idx is None:
        return None

    # Find first assistant after that user
    for j in range(user_idx + 1, len(messages)):
        m = messages[j]
        if (
            isinstance(m, dict)
            and m.get("role") == "assistant"
            and isinstance(m.get("content"), str)
        ):
            user_text = messages[user_idx]["content"].strip()
            assistant_text = m["content"].strip()
            if user_text and assistant_text:
                return {"prompt": user_text, "response": assistant_text}
            return None
    return None


def load_dataset_ultrachat(dataset_name_or_path: str) -> DatasetDict:
    """
    Load the UltraChat 200k dataset (HuggingFaceH4/ultrachat_200k) and keep only the *first turn*
    (first user message and the assistant reply).

    Returns a `DatasetDict` where each split contains:
      - prompt:   first user message content
      - response: first assistant reply content

    Parameters
    ----------
    dataset_name_or_path : str
        Typically "HuggingFaceH4/ultrachat_200k" or a local path.
    data_dir : Optional[str]
        Optional subdirectory (for local paths).
    """
    dataset = load_dataset(dataset_name_or_path)

    # We only keep examples that have a valid first (user, assistant) turn.
    def has_first_turn(example):
        messages = example.get("messages")
        return _extract_first_turn(messages) is not None

    dataset = dataset.filter(has_first_turn, num_proc=4)

    def map_fn(example):
        first = _extract_first_turn(example["messages"])
        # Fallbacks for robustness (shouldn't be hit after filter, but just in case)
        if first is None:
            first = {"prompt": (example.get("prompt") or "").strip(), "response": ""}
        return {"prompt": first["prompt"], "response": first["response"]}

    # Remove original columns for a clean schema (infer from any available split)
    cols_to_remove = None
    for split_name in dataset.keys():
        cols_to_remove = dataset[split_name].column_names
        break

    dataset = dataset.map(map_fn, remove_columns=cols_to_remove, num_proc=4)
    dataset = DatasetDict(
        {
            new: dataset[old]
            for old, new in {
                "train_sft": "train",
                "test_sft": "test",
            }.items()
            if old in dataset
        }
    )
    return dataset


if __name__ == "__main__":
    # Mirrors the style from your previous loaders: resolve path via env helper if available.
    from dllm.utils import resolve_with_base_env

    dataset_name_or_path = resolve_with_base_env(
        "HuggingFaceH4/ultrachat_200k", "BASE_DATASETS_DIR"
    )
    dataset = load_dataset_ultrachat(dataset_name_or_path)
    breakpoint()
