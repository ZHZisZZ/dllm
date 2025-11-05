import os
import functools
from dataclasses import dataclass, field

import transformers
import accelerate

import dllm
from dllm.pipelines import editflow


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = None  # overwrite this
    lm_head_key: str = field(
        default=None,
        metadata={
            "help": (
                "The key to the `lm_head` in the source model for initializing operation heads in the EditFlow model. "
                "Overwrite this when `init_editflow_from_src` = True"
            )
        },
    )
    init_editflow_from_src: bool = field(
        default=True,
        metadata={
            "help": "Whether to initialize EditFlow model from the source model."
        },
    )


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "mlfoundations/dclm-baseline-1.0[train:10_000_000,test:10_000]"
    text_field: str = "text"
    streaming: bool = False
    drop_tail: bool = True
    insert_eos: bool = field(
        default=True,
        metadata={
            "help": "False when adjacent samples from the datasets are semantically coherent."
        },
    )


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = None  # overwrite this
    num_train_epochs: float = 20
    learning_rate: float = 3e-4
    # max_steps: int = 2_000
    per_device_train_batch_size: int = 3
    per_device_eval_batch_size: int = 3
    eval_steps: float = 0.1
    save_steps: float = 0.1
    # EditFlow specific args
    scheduler_cls: str = field(
        default="LinearKappaScheduler",
        metadata={
            "help": (
                "The scheduler class controlling κ(t). "
                "Available options: see `dllm/utils/schedulers/kappa.py`"
            )
        },
    )
    normalize_per_position: bool = field(
        default=True,
        metadata={"help": "Whether to normalize the loss per position."},
    )
    max_w: float = field(
        default=20.0,
        metadata={"help": "The maximum weight (κ'(t) / (1 - κ(t))) for the loss."},
    )
    x0_sampler: str = field(
        default="masks[length:128]",
        metadata={
            "help": (
                "Choose the x0 sampler. "
                "Available options: see `dllm/pipelines/editflow/utils.py`"
            )
        },
    )


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    ef_config_cls: type[transformers.PretrainedConfig],
):
    # necessary when batch does not contain "labels" field
    training_args.label_names = []
    # necessary when batch contains customized fields
    training_args.remove_unused_columns = False
    # necessary for streaming dataset
    training_args.accelerator_config.dispatch_batches = False
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Load base Model and initialize EditFlow Model ---------------------------
    # Create EditFlow model (bf16 init on CUDA)
    ef_cfg = ef_config_cls.from_pretrained(
        model_args.model_name_or_path,
        dtype=model_args.dtype,
        attn_implementation=model_args.attn_implementation,
    )
    with dllm.utils.init_device_context_manager():
        model = transformers.AutoModel.from_config(ef_cfg)
        if model_args.init_editflow_from_src:
            # Load src model config & weights (bf16 on CUDA) for intializing EditFlow model
            src_model = transformers.AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path, dtype=model_args.dtype
            )
            # Initialize EditFlow model from the src model: copies backbone & clones lm_head
            editflow.utils.init_editflow_from_src(
                model, src_model, lm_head_key=model_args.lm_head_key
            )
            del src_model
    model = dllm.utils.load_peft(model, model_args)

    def _no_flops(*args, **kwargs):
        return 0.0

    model.floating_point_ops = _no_flops

    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
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
                drop_tail=data_args.drop_tail),
            batched=True,
            num_proc=None if data_args.streaming else data_args.num_proc,
            remove_columns=dataset["train"].column_names,
        )
        if data_args.streaming: dataset = dataset.shuffle(seed=training_args.seed)

    # ----- Training --------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    dllm.utils.print_main("start training...")
    trainer = editflow.EditFlowTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=editflow.utils.EditFlowCollator(
            tokenizer=tokenizer, x0_sampler=training_args.x0_sampler
        ),
        scheduler=dllm.core.schedulers.make_kappa_scheduler(
            training_args.scheduler_cls
        ),
        normalize_per_position=training_args.normalize_per_position,
        max_w=training_args.max_w,
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )
