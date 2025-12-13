"""
Local users
------------
- 1 GPU (LoRA, useful for testing):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/editflow/llada/pt.py \
        --lora True

- 8 GPUs (DeepSpeed FSDP):
    accelerate launch \
        --config_file scripts/accelerate_configs/fsdp.yaml \
        examples/editflow/llada/pt.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 Node, 8 GPUs (FSDP):
    sbatch --gres=gpu:1 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/editflow/llada/pt.py"

- 24 Nodes, 192 GPUs (FSDP):
    sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/editflow/llada/pt.py"
"""

from dataclasses import dataclass

import transformers

import dllm
from examples.editflow import pt as editflow_pt


@dataclass
class ModelArguments(editflow_pt.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"
    lm_head_key: str = "model.transformer.ff_out"


@dataclass
class DataArguments(editflow_pt.DataArguments):
    dataset_args: str = "mlfoundations/dclm-baseline-1.0[train:10_000_000,test:10_000]"


@dataclass
class TrainingArguments(editflow_pt.TrainingArguments):
    output_dir: str = (
        "models/EditFlow-LLaDA-8B-Base/dclm-baseline-1.0[train:10_000_000,test:10_000]"
    )


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
        ef_config_cls=dllm.pipelines.editflow.EditFlowLLaDAConfig,
    )
