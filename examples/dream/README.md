# Dream

> 📄 Paper: [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487)
> 💻 Code: [github.com/DreamLM/Dream](https://github.com/DreamLM/Dream)

<!-- This directory provides examples for finetuning open-weight Dream models, reproducing Dream by training from scratch on public data (pretraining & finetuning), and batch sampling for generation tasks. -->
This directory provides examples for (1) finetuning open-weight Dream models, (2) pretraining from scratch on public data, (3) interactive inference and (4) evaluation.

## Table of Contents
- [Setup](#setup)
- [Files overview](#files-overview)
- [Training](#training)
    <!-- - [Finetuning Dream-v0-Base-7B](#finetuning-dream-v0-base-7b)
    - [Pretraining from scratch](#pretraining-from-scratch) -->
- [Inference](#inference)
- [Evaluation](#evaluation)

## Setup
> [!IMPORTANT]  
> **Slurm users:** Update `scripts/train.slurm.sh` and `mkdir logps`: see [(optional) Slurm setup](/README.md/#optional-slurm-setup) for details.
>


##  Files overview
```
# tools relevant with Dream
dllm/pipelines/dream
├── __init__.py                     # Package initialization
├── models/
│   ├── configuration_dream.py      # Dream model configuration
│   ├── generation_utils.py         # Diffusion-based generation logic
│   ├── modeling_dream.py           # Core Dream model architecture
│   └── tokenization_dream.py       # Tokenizer implementation for Dream
├── generator.py                    # Inference logic
├── trainer.py                      # Training logic (pretraining and SFT)
└── utils.py                        # Auxiliary utilities and helper functions

# example entry points for training / inference
examples/dream
├── chat.py                         # Interactive inference example
├── generate.py                     # Inference example
├── pt.py                           # Pretraining example
├── README.md                       # Documentation (you are here)
└── sft.py                          # Supervised finetuning example
```
<!-- > [!NOTE]
>  We slightly modified [`modeling_dream.py`](/dllm/pipelines/dream/models/modeling_dream.py) so that the `model.forward()` supports 2-D attention masks. We recommend loading models with `dllm.utils.get_tokenizer`; otherwise `import dllm` before calling `AutoModel.from_pretrained` to ensure the correct models from `dllm` are used. 
> 
> We fixed bugs in `chat_template` and standardize `mask_token` through `dllm.utils.get_tokenizer`. If you use `AutoTokenizer`, keep in mind to set `chat_template` and `mask_token` appropriately yourselves. -->

## Training

<!-- > [!NOTE]
> Here are some useful tips for training.
> - Use a subset of data: `--dataset_args "allenai/tulu-3-sft-mixture[train:10000,test:1000]"`; 
> 
> - Concatenate datasets: `--dataset_args "allenai/tulu-3-sft-mixture|HuggingFaceTB/smoltalk"`;
>
> - Train with LoRA and 4bit quantization: `--load_in_4bit True --lora True`. -->

### Finetuning
For example, to SFT [`Dream-v0-Base-7B`](https://huggingface.co/Dream-org/Dream-v0-Base-7B) for instruction following on 8 GPUs, run:
```shell
accelerate launch \
    --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/dream/sft.py \
    --model_name_or_path "Dream-org/Dream-v0-Base-7B" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/Dream-7B-SFT/tulu-3-sft-mixture" \
    --max_length 1024 \
    --num_train_epochs 4 \
    --learning_rate 2e-5
```
If you are using slurm and want to train across, for example, 2 nodes (16 GPUs total), run:
```shell
sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/dream/sft.py" \
    --model_name_or_path "Dream-org/Dream-v0-Base-7B" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/Dream-7B-SFT/tulu-3-sft-mixture" \
    --max_length 1024 \
    --num_train_epochs 4 \
    --learning_rate 2e-5
```

<!-- **Reproducing [Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Base-7B)**. We tried our best to reproduce Dream-v0-Instruct-7B by finetuning Dream-v0-Base-7B using our training pipeline on the public instruction-following dataset [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture): -->
#### Reproducing [`Dream-v0-Instruct-7B`](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B)
We tried our best to reproduce [`Dream-v0-Instruct-7B`](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) by finetuning [`Dream-v0-Base-7B`](https://huggingface.co/Dream-org/Dream-v0-Base-7B) using our training pipeline on the public instruction-following dataset [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture):

```shell
# preprocessing SFT data (optional, but can avoid redundant preprocessing for multi-node training)
PYTHONPATH=. python dllm/tools/preprocess_sft_dataset.py \
    --model_name_or_path "Dream-org/Dream-v0-Base-7B" \
    --sft_map_fn_path "examples.dream.sft.sft_map_fn" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "data/sft/dream/tulu-3-sft-mixture" \
    --num_proc 64

# train on 24*8=192 A100s with FSDP, take about 8 hours
sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/dream/sft.py" \
    --model_name_or_path "Dream-org/Dream-v0-Base-7B" \
    --dataset_args "data/sft/dream/tulu-3-sft-mixture" \
    --load_preprocessed_data True \
    --output_dir "models/Dream-7B-SFT-tulu3-fsdp-bs4-len2048-ep5-lr1e-5" \
    --max_length 2048 \
    --truncation "right" \
    --group_by_length True \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 2 \
    --eval_on_start False \
    --eval_steps 0.1 \
    --save_steps 0.05
```
[TODO] Training curves are on Wandb; checkpoints with evaluation results are available on Hugging Face. See the [Evaluation](#evaluation) section below for evaluation instructions.

### Pretraining
<!-- > [!NOTE]
> This is an educational example demonstrating how to reproduce Dream pretraining and finetuning on public data. We do not guarantee performance comparable to the official Dream models. -->

Pretrain on [`mlfoundations/dclm-baseline-1.0`](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) from scratch using 192 GPUs (24x8) and FSDP:
```shell
sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/dream/pt.py" \
    --model_name_or_path "Dream-org/Dream-v0-Base-7B" \
    --dataset_args "mlfoundations/dclm-baseline-1.0" \
    --output_dir "models/Dream-7B-PT/dclm-baseline-1.0" \
    --max_length 1024 \
    --max_steps 2000 \
    --learning_rate 3e-4
```

## Inference
We support batch inference for standard generation and infilling:
<!-- See [`examples/dream/generate.py`](/examples/dream/generate.py) for a full example: -->
```shell
python examples/dream/generate.py --model_name_or_path "Dream-org/Dream-v0-Instruct-7B"
```
We also support interactive multi-turn dialogue with visualization:
```shell
python examples/dream/chat.py --model_name_or_path "Dream-org/Dream-v0-Instruct-7B"
```

## Evaluation  
> Read [(optional) Evaluation setup](/README.md/#optional-evaluation-setup) before running evaluation. 

For example, to evaluate [`Dream-v0-Instruct-7B`](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) on [`MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) using 4 GPUs, run:
```shell
# Use model_args to adjust the generation arguments for evalution.
accelerate launch --num_processes 4 \
    dllm/pipelines/dream/eval.py \
    --tasks "mmlu_pro" \
    --model "dream" \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,mc_num=1,max_new_tokens=128,max_length=128,steps=128,temperature=0.1,top_p=0.9,add_bos_token=true,escape_until=true"
```

To automatically evaluate [`Dream-v0-Base-7B`](https://huggingface.co/Dream-org/Dream-v0-Base-7B) and [`Dream-v0-Instruct-7B`](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) on all benchmarks, run:
```shell
bash examples/llada/eval.sh --model_name_or_path "Dream-org/Dream-v0-Instruct-7B" --instruct True
bash examples/llada/eval.sh --model_name_or_path "Dream-org/Dream-v0-Base-7B" --instruct False
```

> [!IMPORTANT]  
> 1. *Evaluated* results are obtained from our own evaluation framework, while *Reported* results come from the original paper.  
> 2. All evaluation settings follow the configurations in the [Dream](https://github.com/DreamLM/Dream) repository, with minor adjustments for compatibility.  
> 3. Full evaluation results will be released soon.



### Evaluation Method of Dream-8B-Base

| Evaluation Type | MMLU | BBH | ARC-C | ARC-E | Hellaswag | WinoGrande | PIQA | GSM8K | Math | GPQA | HumanEval | MBPP | RACE |
|:----------------|:----:|:---:|:-----:|:-----:|:-----------:|:------------:|:----:|:-----:|:----:|:----:|:-----------:|:----:|:------:|
| Reported | 0.695 | 0.579 | 0.599 | 0.839 | 0.733 | 0.748 | 0.758 | 0.772 | 0.396 | 0.366 | 0.579 | 0.562 | 0.447 |
| Evaluated | - | – | - | - | - | - | - | 0.696 | – | - | - | - | 0.304 |


### Evaluation Method of Dream-8B-Instruct

| Evaluation Type | MMLU | MMLU-Pro | GSM8K | Math | GPQA | HumanEval | MBPP |
|:----------------|:----:|:---------:|:-----:|:----:|:----:|:-----------:|:----:|
| Reported | 0.670 | 0.433 | 0.810 | 0.392 | 0.330 | 0.555 | 0.588 |
| Evaluated | – | – | 0.826 | – | - | 0.591 | – |


