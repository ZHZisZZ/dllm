from dataclasses import dataclass

import transformers

import dllm
from examples.editflow import pt as editflow_pt


@dataclass
class ModelArguments(editflow_pt.ModelArguments):
    model_name_or_path: str = "answerdotai/ModernBERT-large"
    lm_head_key: str = "decoder"

@dataclass
class DataArguments(editflow_pt.DataArguments):
    dataset_args: str = "Trelis/tiny-shakespeare"
    text_field: str = "Text"
    max_length: int = 128
    streaming: bool = False
    drop_tail: bool = True
    insert_eos: bool = False


@dataclass
class TrainingArguments(editflow_pt.TrainingArguments):
    output_dir: str = "models/EditFlow/ModernBERT-large/tiny-shakespeare"
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
    editflow_pt.train(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        ef_config_cls=dllm.pipelines.editflow.EditFlowModernBertConfig,
    )
