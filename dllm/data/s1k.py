from datasets import DatasetDict, load_dataset

# messages = [
#     {"role": "user", "content": "Solve 13 * 17"},
#     {
#         "role": "assistant",
#         "reasoning_content": "We need to multiply 13 and 17 step by step.",
#         "content": "13 * 17 = 221."
#     }
# ]


def load_dataset_s1k(dataset_name_or_path: str) -> DatasetDict:

    dataset = load_dataset(dataset_name_or_path)

    def map_fn(example):

        return {
            "messages": [
                {"role": "user", "content": example["question"]},
                {
                    "role": "assistant",
                    "reasoning_content": example["thinking_trajectories"][0],
                    "content": example["attempt"],
                },
            ]
        }

    dataset = dataset.map(
        map_fn, remove_columns=dataset["train"].column_names, num_proc=4
    )
    return dataset


if __name__ == "__main__":
    from dllm.utils import resolve_with_base_env

    dataset_name_or_path = resolve_with_base_env(
        "simplescaling/s1K", "BASE_DATASETS_DIR"
    )
    dataset = load_dataset_s1k(dataset_name_or_path)
    breakpoint()
