from dataclasses import dataclass

import transformers

import dllm
from examples.editflow import sft as editflow_sft


@dataclass
class ModelArguments(editflow_sft.ModelArguments):
    model_name_or_path: str = "answerdotai/ModernBERT-large"
    lm_head_key: str = "decoder"


@dataclass
class DataArguments(editflow_sft.DataArguments):
    dataset_args: str = "tatsu-lab/alpaca"
    max_length: int = 512


@dataclass
class TrainingArguments(editflow_sft.TrainingArguments):
    output_dir: str = "models/EditFlow/ModernBERT-large/alpaca"
    num_train_epochs: float = 20
    learning_rate: float = 3e-4
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    eval_steps: float = 0.1
    save_steps: float = 0.1
    x0_sampler: str = "masks[length:64]"


if __name__ == "__main__":
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    editflow_sft.train(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        ef_config_cls=dllm.pipelines.editflow.EditFlowModernBertConfig,
    )
