"""
Local users
------------
- 1 GPU (LoRA, useful for testing):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/editflow/dream/adapt.py \
        --lora True

- 8 GPUs (FSDP):
    accelerate launch \
        --config_file scripts/accelerate_configs/fsdp.yaml \
        examples/editflow/dream/adapt.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 Node, 8 GPUs (FSDP):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/editflow/dream/adapt.py"

- 2 Nodes, 16 GPUs (FSDP):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/editflow/dream/adapt.py"
"""

from dataclasses import dataclass

import torch
import transformers

import dllm
from examples.editflow import sft as editflow_sft


@dataclass
class ModelArguments(editflow_sft.ModelArguments):
    model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
    lm_head_key: str = "lm_head"
    init_editflow_from_src: bool = True


@dataclass
class DataArguments(editflow_sft.DataArguments):
    dataset_args: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]"


@dataclass
class TrainingArguments(editflow_sft.TrainingArguments):
    output_dir: str = (
        "models/EditFlow-Dream-7B-Instruct-Adapt/tulu-3-sft-mixture[train:10000,test:1000]"
    )


if __name__ == "__main__":
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    dllm.utils.initial_training_setup(model_args, data_args, training_args)
    # Create EditFlow model (bf16 init on CUDA)
    ef_cfg = dllm.pipelines.editflow.EditFlowDreamConfig.from_pretrained(
        model_args.model_name_or_path
    )
    with dllm.utils.init_device_context_manager():
        model = transformers.AutoModel.from_config(ef_cfg, dtype=torch.bfloat16)
        # Initialize EditFlow model from the src model: copies backbone & clones lm_head
        if model_args.init_editflow_from_src:
            src_model = transformers.AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path, dtype=torch.bfloat16
            )
            dllm.pipelines.editflow.utils.init_editflow_from_src(
                model, src_model, lm_head_key=model_args.lm_head_key
            )
            del src_model
    model = dllm.utils.load_peft(model, model_args)

    editflow_sft.train(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        model=model,
    )
