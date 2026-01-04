"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/a2d/bd3lm/pt.py

- 8 GPUs (ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/zero2.yaml \
        examples/a2d/bd3lm/pt.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 Node, 8 GPUs (ZeRO-2):
    sbatch --gres=gpu:1 scripts/train.slurm.sh \
        --accelerate_config "zero2" \
        --script_path "examples/a2d/bd3lm/pt.py"

- 2 Nodes, 16 GPUs (ZeRO-2):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "zero2" \
        --script_path "examples/a2d/bd3lm/pt.py"
"""

import functools
import os
from dataclasses import dataclass, field

import accelerate
import transformers

import dllm

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "models/a2d/Qwen3-0.6B"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "Trelis/tiny-shakespeare"
    text_field: str = "Text"
    max_length: int = 128
    streaming: bool = False
    drop_tail: bool = True
    insert_eos: bool = field(
        default=True,
        metadata={
            "help": "False when adjacent samples from the datasets are semantically coherent."
        },
    )
    load_preprocessed_data: bool = False


@dataclass
class TrainingArguments(dllm.core.trainers.BD3LMTrainer.BD3LMConfig):
    output_dir: str = "models/a2d/Qwen3-0.6B/bd3lm/tiny-shakespeare"
    num_train_epochs: int = 20
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    # bd3lm
    block_size: int = 32


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    model = dllm.utils.get_model(model_args=model_args)
    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_pt_dataset(
            data_args.dataset_args,
            streaming=data_args.streaming,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
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
                **(
                    {}
                    if data_args.streaming
                    else {"desc": "Mapping dataset to PT format"}
                ),
            )
        if data_args.streaming:
            dataset = dataset.shuffle(seed=training_args.seed)

    # ----- Training --------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start training...")
    trainer = dllm.core.trainers.BD3LMTrainer(
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
