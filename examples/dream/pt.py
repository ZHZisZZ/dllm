"""
Local users
------------
- 1 GPU (4bit quant & LoRA, useful for testing):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/dream/pt.py \
        --load_in_4bit True --lora True

- 8 GPUs (FSDP):
    accelerate launch \
        --config_file scripts/accelerate_configs/fsdp.yaml \
        examples/dream/pt.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 24 Nodes, 192 GPUs (FSDP):
    sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/dream/pt.py"
"""

import functools
import os
from dataclasses import dataclass, field

import accelerate
import torch
import transformers

import dllm
from dllm.pipelines import dream

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "Dream-org/Dream-v0-Base-7B"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "mlfoundations/dclm-baseline-1.0[train:10_000_000,test:10_000]"
    text_field: str = "text"
    streaming: bool = True
    drop_tail: bool = True
    insert_eos: bool = field(
        default=True,
        metadata={
            "help": "False when adjacent samples from the datasets are semantically coherent."
        },
    )
    random_length_ratio: float = field(
        default=0.01,
        metadata={
            "help": (
                "The probability of randomly cut sequences during training. "
                "See https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md."
            )
        },
    )


@dataclass
class TrainingArguments(dllm.pipelines.dream.DreamTrainer.DreamConfig):
    output_dir: str = (
        "models/Dream-v0-Base-7B/dclm-baseline-1.0[train:10_000_000,test:10_000]"
    )
    max_steps: int = 100_000
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    # Dream PT specific args
    # Note: Since Dream’s pretraining recipe is not public,
    # this is only a reference implementation following LLaDA’s data processing approach.
    loss_weight_type: str = field(
        default="cart[geo_p:0.3]",
        metadata={
            "help": (
                "The loss weight type. "
                "See https://github.com/DreamLM/Dream/blob/main/src/trainer/config/sft_trainer.yaml."
            )
        },
    )


def train():
    # ----- Parse & setup --------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # necessary for streaming dataset
    if data_args.streaming:
        training_args.accelerator_config.dispatch_batches = False
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ---------------------------------------------------------------
    # initialize model weights from scratch
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    with dllm.utils.init_device_context_manager():
        model = transformers.AutoModel.from_config(config, dtype=torch.bfloat16)

    # ----- Tokenizer -----------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    # ----- Optional PEFT: LoRA -------------------------------------------------
    model = dllm.utils.load_peft(model=model, model_args=model_args)

    # ----- Dataset -------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_pt_dataset(
            data_args.dataset_args,
            streaming=data_args.streaming,
        )
        dataset = dataset.map(
            functools.partial(
                dllm.utils.tokenize_and_group,
                tokenizer=tokenizer,
                text_field=data_args.text_field,
                seq_length=data_args.max_length,
                insert_eos=data_args.insert_eos,
                drop_tail=data_args.drop_tail,
            ),
            batched=True,
            remove_columns=dataset["train"].column_names,
            **({} if data_args.streaming else {"num_proc": data_args.num_proc}),
            **({} if data_args.streaming else {"desc": "Mapping dataset to PT format"}),
        )
        if data_args.streaming:
            dataset = dataset.shuffle(seed=training_args.seed)

    # ----- Training --------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start training...")
    trainer = dream.DreamTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            padding=True,
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
