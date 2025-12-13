"""
Local users
------------
- 1 GPU (LoRA, useful for testing):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/editflow/llada/sft.py \
        --lora True

- 8 GPUs (FSDP):
    accelerate launch \
        --config_file scripts/accelerate_configs/fsdp.yaml \
        examples/editflow/llada/sft.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 Node, 8 GPUs (FSDP):
    sbatch --gres=gpu:1 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/editflow/llada/sft.py"

- 2 Nodes, 16 GPUs (FSDP):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/editflow/llada/sft.py"
"""

from dataclasses import dataclass

import transformers

from examples.editflow import sft as editflow_sft


@dataclass
class ModelArguments(editflow_sft.ModelArguments):
    model_name_or_path: str = (
        "models/EditFlow-LLaDA-8B-Base/dclm-baseline-1.0[train:10_000_000,test:10_000]/checkpoint-final"
    )


@dataclass
class DataArguments(editflow_sft.DataArguments):
    dataset_args: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]"


@dataclass
class TrainingArguments(editflow_sft.TrainingArguments):
    output_dir: str = (
        "models/EditFlow-LLaDA-8B-Instruct-SFT/tulu-3-sft-mixture[train:10000,test:1000]"
    )


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
    )
