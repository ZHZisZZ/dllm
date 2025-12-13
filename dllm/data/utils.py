import re

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
    load_from_disk,
)

from dllm.utils.utils import get_default_logger, parse_spec, resolve_with_base_env

logger = get_default_logger(__name__)


def load_sft_dataset(
    dataset_args: str, load_preprocessed_data: bool = False
) -> DatasetDict:
    """
    Examples of dataset_args:
      - "tatsu-lab/alpaca"
      - "OpenCoder-LLM/opc-sft-stage2[name:educational_instruct,lang:python]"
      - "tatsu-lab/alpaca[train:5000]"
      - "tatsu-lab/alpaca[train:5000] + HuggingFaceH4/ultrachat_200k[train:5000]"
    """
    from dllm.data.alpaca import load_dataset_alpaca
    from dllm.data.opc import load_dataset_opc_sft

    specs = [p.strip() for p in re.split(r"[|+]", dataset_args) if p.strip()]
    all_parts = []

    for raw in specs:
        dataset_name_or_path, kvs = parse_spec(raw)

        dataset_name_or_path = resolve_with_base_env(
            dataset_name_or_path, "BASE_DATASETS_DIR"
        )

        if load_preprocessed_data:
            logger.info("Load preprocessed data from disk.")
            ds = load_from_disk(dataset_name_or_path)
        # Implement your customized dataset here
        elif _match(dataset_name_or_path, "tatsu-lab/alpaca"):
            ds = load_dataset_alpaca(dataset_name_or_path)
        elif _match(dataset_name_or_path, "allenai/tulu-3-sft-mixture"):
            ds = load_dataset(dataset_name_or_path)
            ds = ds["train"].train_test_split(test_size=0.05, seed=42)
        elif _match(dataset_name_or_path, "HuggingFaceTB/smoltalk"):
            name = kvs.pop("name", "all")
            ds = load_dataset(dataset_name_or_path, name=name)
        elif _match(dataset_name_or_path, "OpenCoder-LLM/opc-sft-stage1") or _match(
            dataset_name_or_path, "OpenCoder-LLM/opc-sft-stage2"
        ):
            name = kvs.pop("name", None)
            lang = kvs.pop("lang", None)
            ds = load_dataset_opc_sft(dataset_name_or_path, name=name, lang=lang)
        elif _match(dataset_name_or_path, "HuggingFaceH4/ultrachat_200k"):
            ds = load_dataset(dataset_name_or_path)
            ds = DatasetDict({"train": ds["train_sft"], "test": ds["test_sft"]})
        else:
            ds = load_dataset(dataset_name_or_path)

        # Normalize to DatasetDict and apply per-split limits
        ds = _ensure_datasetdict(ds)
        ds = _truncate_datasetdict(ds, kvs)
        all_parts.append(ds)

    # If only one part, return as DatasetDict
    if len(all_parts) == 1:
        return _ensure_datasetdict(all_parts[0])

    # Merge all parts into a single DatasetDict
    merged = all_parts[0]
    for part in all_parts[1:]:
        merged = _merge_datasetdicts(merged, part)
    return _ensure_datasetdict(merged)


def load_pt_dataset(
    dataset_args: str, streaming: bool = True, load_preprocessed_data: bool = False
) -> DatasetDict | IterableDatasetDict:
    """
    Examples of dataset_args:
      - "mlfoundations/dclm-baseline-1.0"
      - "OpenCoder-LLM/opc-fineweb-code-corpus"
      - "OpenCoder-LLM/opc-fineweb-math-corpus"
      - "OpenCoder-LLM/opc-annealing-corpus[lang:python]"
      - "wikitext[name:wikitext-103-v1}]"
    """
    from dllm.data.opc import load_dataset_opc_annealing

    specs = [p.strip() for p in re.split(r"[|+]", dataset_args) if p.strip()]
    if not specs:
        raise ValueError("Empty dataset_args for load_pt_dataset.")

    # ---------- Shared loader (only differs by streaming flag) ----------
    def _load_base_dataset(
        raw: str, *, streaming: bool
    ) -> tuple[DatasetDict | IterableDatasetDict, dict, str]:
        """
        Returns: (base, kvs, dataset_name_or_path)
        - Pops 'name' from kvs when applicable (e.g., wikitext).
        - Applies identical matching logic for both streaming/non-streaming.
        """
        dataset_name_or_path, kvs = parse_spec(raw)
        dataset_name_or_path = resolve_with_base_env(
            dataset_name_or_path, "BASE_DATASETS_DIR"
        )
        name = kvs.pop("name", None)

        if load_preprocessed_data:
            base = load_from_disk(dataset_name_or_path)
        elif _match(dataset_name_or_path, ["OpenCoder-LLM/opc-annealing-corpus"]):
            lang = kvs.pop("lang", None)
            base = load_dataset_opc_annealing(
                dataset_name_or_path, name=name, lang=lang, streaming=streaming
            )
        else:
            base = load_dataset(dataset_name_or_path, name=name, streaming=streaming)

        return base, kvs, dataset_name_or_path

    # ---------- Streaming path ----------
    def _load_one_streaming_spec(raw: str) -> IterableDatasetDict:
        base, kvs, _ = _load_base_dataset(raw, streaming=True)
        ds = _ensure_iterabledatasetdict(base)
        ds = _truncate_iterabledatasetdict(base, kvs)
        return ds

    # ---------- Non-streaming path (mirror load_sft_dataset; NO shuffle) ----------
    def _load_one_nonstreaming_spec(raw: str) -> DatasetDict:
        base, kvs, _ = _load_base_dataset(raw, streaming=False)
        ds = _ensure_datasetdict(base)  # normalize
        ds = _truncate_datasetdict(ds, kvs)  # apply limits (train/test/...)
        return ds

    # ---------- Load & Merge ----------
    if streaming:
        logger.info("Loading dataset in streaming mode.")
        parts = [_load_one_streaming_spec(raw) for raw in specs]
        merged = parts[0]
        for p in parts[1:]:
            merged = _merge_iterabledatasetdicts(merged, p)
        # repeat streaming dataset infinitely
        merged = IterableDatasetDict(
            {k: (v.repeat(None) if k == "train" else v) for k, v in merged.items()}
        )
        return merged
    else:
        logger.info("Loading dataset in non-streaming mode.")
        parts = [_load_one_nonstreaming_spec(raw) for raw in specs]
        if len(parts) == 1:
            return _ensure_datasetdict(parts[0])
        merged = parts[0]
        for p in parts[1:]:
            merged = _merge_datasetdicts(merged, p)
        return _ensure_datasetdict(merged)


def _truncate_split(split_data, n: int):
    if n is None:
        return split_data
    try:
        if hasattr(split_data, "select"):
            # Hugging Face Dataset path
            total = getattr(split_data, "num_rows", None)
            if total is None:
                # some Dataset types expose len(...)
                total = len(split_data)
            idx = list(range(min(n, total)))
            return split_data.select(idx)
    except Exception:
        pass
    try:
        return split_data[:n]
    except Exception:
        # Last resort: iterate
        return type(split_data)(item for i, item in enumerate(split_data) if i < n)


def _truncate_datasetdict(ds, limits: dict):
    """
    Ensure and return a DatasetDict, truncating splits mentioned in `limits`.

    If the dataset has only one split but `limits` contains multiple
    target splits (e.g., train/test), we create each requested split
    independently by truncating the original single split.
    """
    ds = _ensure_datasetdict(ds)
    split_names = list(ds.keys())

    # ---- Single-split case: create requested splits from the same source ----
    if len(split_names) == 1:
        base = ds[split_names[0]]
        out = {}

        for split_name, n in limits.items():
            if n is None:
                continue
            # Each target split is truncated independently from the same base
            out[split_name] = _truncate_split(base, n)

        # If no valid limits are given, return unchanged dataset
        if not out:
            return ds

        return DatasetDict(out)

    # ---- Multi-split case: truncate only the splits mentioned in limits ----
    out = {}
    for split, data in ds.items():
        n = limits.get(split)
        out[split] = _truncate_split(data, n) if n is not None else data

    return DatasetDict(out)


def _concat_splits(a, b):
    """
    Concatenate two split objects (prefer 🤗 datasets).
    """
    if a is b:
        return a
    if a is None:
        return b
    if b is None:
        return a

    # Prefer datasets' concatenate_datasets when both are Datasets
    try:
        from datasets import concatenate_datasets

        if isinstance(a, Dataset) and isinstance(b, Dataset):
            return concatenate_datasets([a, b])
    except Exception:
        pass

    # Fallbacks
    try:
        return a + b
    except Exception:
        pass
    try:
        return type(a)(list(a) + list(b))
    except Exception:
        pass

    raise TypeError(
        f"Cannot concatenate split objects of types {type(a)} and {type(b)}"
    )


def _merge_datasetdicts(d1, d2):
    """
    Merge two DatasetDict-like mappings by concatenating splits present in either.
    Always returns a DatasetDict.
    """
    d1 = _ensure_datasetdict(d1)
    d2 = _ensure_datasetdict(d2)
    all_splits = set(d1.keys()) | set(d2.keys())
    out = {}
    for split in all_splits:
        a = d1.get(split, None)
        b = d2.get(split, None)
        if a is None:
            out[split] = b
        elif b is None:
            out[split] = a
        else:
            out[split] = _concat_splits(a, b)
    return DatasetDict(out)


def _ensure_datasetdict(ds):
    """
    Normalize various loader outputs into a DatasetDict.
    - If loader returns a DatasetDict, return as is.
    - If loader returns a mapping (e.g., dict of splits), wrap into DatasetDict.
    - If loader returns a single Dataset/list/etc., assume it's 'train'.
    """
    if isinstance(ds, DatasetDict):
        return ds
    if isinstance(ds, dict):
        # Try to convert each split value to a Dataset if they aren't already.
        # If they are already Datasets, DatasetDict will accept them directly.
        return DatasetDict(ds)
    # Single split -> assume train
    return DatasetDict({"train": ds})


def _match(name: str, needle) -> bool:
    """
    Returns True if `name` matches any of the provided needles.
    Accepts a single string or a list/tuple of strings.
    Match condition: name endswith(needle) or needle in name.
    """
    if isinstance(needle, (list, tuple)):
        return any(name.endswith(n) or n in name for n in needle)
    return name.endswith(needle) or needle in name


def _truncate_iterabledatasetdict(
    base: IterableDatasetDict | dict,
    limits: dict,
    # dataset_name_or_path: str,
) -> IterableDatasetDict:
    """
    Apply train/test limits to an IterableDatasetDict.

    - If no train/test limits are provided, return the dataset unchanged.
    - If there is only one split, derive train/test (or just train / just test)
      from that single split.
    - If there are multiple splits, require the corresponding named split(s).
    """
    # Normalize to IterableDatasetDict
    if not isinstance(base, IterableDatasetDict):
        base = IterableDatasetDict(base)

    n_train = limits.get("train")
    n_test = limits.get("test")

    # No limits → return as is
    if n_train is None and n_test is None:
        return base

    split_names = list(base.keys())
    single_split = len(split_names) == 1
    single_split_name = split_names[0] if single_split else None

    # Both train and test requested
    if n_train is not None and n_test is not None:
        if single_split:
            # Use a single underlying stream, split into test then train
            stream = base[single_split_name]
            head = stream.take(n_train + n_test)
            test = head.take(n_test)
            train = head.skip(n_test).take(n_train)
            return IterableDatasetDict({"train": train, "test": test})

        # Multi-split: require explicit train/test splits
        if "train" not in base or "test" not in base:
            raise ValueError("require 'train' and 'test' splits for train+test limits.")
        train = base["train"].take(n_train)
        test = base["test"].take(n_test)
        return IterableDatasetDict({"train": train, "test": test})

    # Only train requested
    if n_train is not None:
        if single_split:
            train = base[single_split_name].take(n_train)
        else:
            if "train" not in base:
                raise ValueError("missing 'train' split for train limit.")
            train = base["train"].take(n_train)
        return IterableDatasetDict({"train": train})

    # Only test requested
    if n_test is not None:
        if single_split:
            test = base[single_split_name].take(n_test)
        else:
            if "test" not in base:
                raise ValueError("missing 'test' split for test limit.")
            test = base["test"].take(n_test)
        return IterableDatasetDict({"test": test})

    # Fallback (should not be reached)
    return base


def _concat_iterabledatasets(parts: list[IterableDataset]) -> IterableDataset:
    """
    Concatenate IterableDatasets sequentially without materialization.
    Preserves streaming nature; supports downstream .take()/.skip()/.shuffle().
    """
    if not parts:
        raise ValueError("No IterableDatasets to concatenate.")
    # Try to reuse features from the first dataset when available
    features = getattr(parts[0], "features", None)

    def _gen():
        for ds in parts:
            yield from ds

    return IterableDataset.from_generator(_gen, features=features)


def _ensure_iterabledatasetdict(obj) -> IterableDatasetDict:
    if isinstance(obj, IterableDatasetDict):
        return obj
    if isinstance(obj, dict):
        return IterableDatasetDict(obj)
    # Single stream -> assume train
    return IterableDatasetDict({"train": obj})


def _merge_iterabledatasetdicts(
    d1: IterableDatasetDict, d2: IterableDatasetDict
) -> IterableDatasetDict:
    """
    Merge by concatenating any overlapping splits (streaming-safe).
    """
    d1 = _ensure_iterabledatasetdict(d1)
    d2 = _ensure_iterabledatasetdict(d2)
    all_splits = set(d1.keys()) | set(d2.keys())
    out = {}
    for split in all_splits:
        a = d1.get(split, None)
        b = d2.get(split, None)
        if a is None:
            out[split] = b
        elif b is None:
            out[split] = a
        else:
            out[split] = _concat_iterabledatasets([a, b])
    return IterableDatasetDict(out)


def _truncate_stream(ds: IterableDataset, n: int | None) -> IterableDataset:
    if n is None:
        return ds
    return ds.take(n)


if __name__ == "__main__":
    breakpoint()
