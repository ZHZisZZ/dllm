from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)

from dllm.data.utils import (
    _ensure_datasetdict,
    _ensure_iterabledatasetdict,
    _merge_datasetdicts,
    _merge_iterabledatasetdicts,
)


def load_dataset_opc_sft(
    dataset_name_or_path: str, name: str | None = None, lang: str | None = None
) -> DatasetDict:
    """
    Load OpenCoder OPC SFT dataset(s) and produce a DatasetDict with a train/test split.
    - If `name` is provided: load that specific config.
    - If `name` is None: load *all* available configs and concatenate them.
    """

    def _map_to_messages(ds: Dataset) -> Dataset:
        def map_fn(example):
            return {
                "messages": [
                    {"role": "user", "content": example["instruction"]},
                    {"role": "assistant", "content": example["output"]},
                ]
            }

        # Remove all original columns after mapping
        remove_cols = ds.column_names
        return ds.map(map_fn, remove_columns=remove_cols, num_proc=4)

    def _load_one_config(dataset_name_or_path: str, cfg_name: str) -> Dataset:
        ds = load_dataset(dataset_name_or_path, cfg_name, split="train")
        return _map_to_messages(ds)

    if name is not None:
        train_ds = _load_one_config(dataset_name_or_path, name)
    else:
        # Enumerate and load all configs, then concatenate
        cfgs: list[str] = get_dataset_config_names(dataset_name_or_path)
        if not cfgs:
            raise ValueError(f"No configs found for dataset: {dataset_name_or_path}")
        parts = [_load_one_config(dataset_name_or_path, c) for c in cfgs]
        train_ds = concatenate_datasets(parts)

    # Final split
    ds_dict = train_ds.train_test_split(test_size=0.05, seed=42)
    if lang is not None:
        ds_dict = ds_dict.filter(lambda row: lang in row["messages"][1]["content"])

    return DatasetDict(ds_dict)


def load_dataset_opc_annealing(
    dataset_name_or_path: str,
    name: str | None = None,
    lang: str | None = None,
    streaming: bool = True,
) -> DatasetDict:
    """
    Load OpenCoder OPC annealing dataset(s).

    Args:
        dataset_name_or_path: Path or name of the dataset.
        name: Specific config name to load, or None to load all configs.
        lang: Language filter (e.g., "python"), or None for all languages.
        streaming: If True, return IterableDatasetDict; if False, return DatasetDict.

    Returns:
        Dataset dictionary with train split.
    """

    def _load_one_config(_name):
        ds = load_dataset(
            dataset_name_or_path, _name, split="train", streaming=streaming
        )
        if lang:
            if _name in ["synthetic_code_snippet", "algorithmic_corpus"]:
                ds = ds.filter(lambda row: row["lang"] == lang)
            elif _name in ["synthetic_qa"]:
                ds = ds.filter(lambda row: row["program_lang"] == lang)
            else:
                raise NotImplementedError
        # return IterableDatasetDict({"train": ds})
        if streaming:
            return _ensure_iterabledatasetdict(ds)
        return _ensure_datasetdict(ds)

    if name is not None:
        return _load_one_config(name)

    if streaming:
        parts = [
            _load_one_config(name)
            for name in get_dataset_config_names(dataset_name_or_path)
        ]
        merged = parts[0]
        for p in parts[1:]:
            merged = _merge_iterabledatasetdicts(merged, p)
        return merged
    else:
        parts = [
            _load_one_config(name)
            for name in get_dataset_config_names(dataset_name_or_path)
        ]
        if len(parts) == 1:
            return _ensure_datasetdict(parts[0])
        merged = parts[0]
        for p in parts[1:]:
            merged = _merge_datasetdicts(merged, p)
        return _ensure_datasetdict(merged)


if __name__ == "__main__":
    from dllm.utils import resolve_with_base_env

    dataset_name_or_path = resolve_with_base_env(
        "OpenCoder-LLM/opc-sft-stage1", "BASE_DATASETS_DIR"
    )
    # If you want a specific config:
    dataset_edu = load_dataset_opc_sft(dataset_name_or_path, "realuser_instruct")
    # Otherwise, all configs concatenated:
    dataset_all = load_dataset_opc_sft(dataset_name_or_path, None)
    dataset_all_python = load_dataset_opc_sft(dataset_name_or_path, None, "python")
    breakpoint()

    # streaming = True
    # dataset_name_or_path = resolve_with_base_env(
    #     "OpenCoder-LLM/opc-annealing-corpus", "BASE_DATASETS_DIR"
    # )
    # # If you want a specific config:
    # dataset_alg_all = load_dataset_opc_annealing(dataset_name_or_path, "algorithmic_corpus")
    # dataset_alg_python = load_dataset_opc_annealing(dataset_name_or_path, "algorithmic_corpus", "python")
    # # Otherwise, all configs concatenated:
    # dataset_all_python = load_dataset_opc_annealing(dataset_name_or_path, None, "python")
    # dataset_all_all = load_dataset_opc_annealing(dataset_name_or_path, None)
    # breakpoint()
