"""
DiffusionGemma SFT.

Examples:
------------
- 1 GPU smoke / QLoRA:
    PYTHONNOUSERSITE=1 PYTHONPATH=. accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/diffusiongemma/sft.py \
        --load_in_4bit True --lora True \
        --dataset_args "path/to/chinese-sft-jsonl[train:1000,test:100]"

- Train the native MoE expert tensors and router projection only:
    PYTHONNOUSERSITE=1 PYTHONPATH=. accelerate launch \
        --config_file scripts/accelerate_configs/fsdp.yaml \
        examples/diffusiongemma/sft.py \
        --dataset_args "path/to/chinese-sft-jsonl[train:10000,test:1000]" \
        --trainable_parameter_patterns ".*\\.language_model\\.layers\\.[0-9]+\\.experts\\.(gate_up_proj|down_proj)$,.*\\.language_model\\.layers\\.[0-9]+\\.router\\.proj\\.weight$"

The dataset must provide a `messages` column compatible with the tokenizer chat
template, for example:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import os
from dataclasses import dataclass, field

import accelerate
import transformers

import dllm
from dllm.pipelines import diffusiongemma

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "/mnt/weka/shrd/research/model/diffusiongemma-26B-A4B-it"
    trainable_parameter_patterns: str | None = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated regexes for base-model parameters to unfreeze. "
                "When LoRA is disabled, all non-matching parameters are frozen. "
                "Use this for native DiffusionGemma expert tensors, which are "
                "nn.Parameter tensors rather than LoRA-compatible Linear modules."
            )
        },
    )


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]"
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on prompt tokens."},
    )


@dataclass
class TrainingArguments(diffusiongemma.DiffusionGemmaTrainerConfig):
    output_dir: str = (
        ".models/diffusiongemma-26B-A4B-it/tulu-3-sft-mixture[train:10000,test:1000]"
    )
    group_by_length: bool = True
    num_train_epochs: float = 1
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    canvas_length: int = 256
    save_final_model: bool = True


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    model = dllm.utils.get_model(model_args=model_args)
    dllm.utils.apply_trainable_parameter_patterns(model, model_args.trainable_parameter_patterns)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
            dataset = dataset.map(
                lambda row: dllm.utils.default_sft_map_fn(
                    row,
                    tokenizer=tokenizer,
                    mask_prompt_loss=data_args.mask_prompt_loss,
                ),
                num_proc=data_args.num_proc,
                desc="Mapping dataset to SFT format",
            )
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    accelerate.PartialState().wait_for_everyone()
    logger.info("Start DiffusionGemma SFT...")
    trainer = diffusiongemma.DiffusionGemmaTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=-100,
        ),
    )
    trainer.train()
    if training_args.save_final_model:
        trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
        trainer.processing_class.save_pretrained(
            os.path.join(training_args.output_dir, "checkpoint-final")
        )


if __name__ == "__main__":
    train()
