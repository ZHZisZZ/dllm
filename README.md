<h1 align="center">dLLM</h1>

<p align="center">
Simple Diffusion Language Modeling
</p>

<p align="center">
<img
  src="assets/logo.gif"
  alt="dLLM logo">
</p>


## Overview
**dLLM** is a library that unifies the training and evaluation of **diffusion language models**, bringing transparency and reproducibility to the entire development pipeline:

 <!-- and [RND1](https://www.radicalnumerics.ai/assets/rnd1_report.pdf) -->

- dLLM provides scalable training pipelines (inspired by [`transformers`](https://github.com/huggingface/transformers/blob/main/src/transformers) [Trainer](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py)), with support for [LoRA](https://github.com/huggingface/peft), [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) and [FSDP](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) and beyond.

- dLLM provides unified evaluation pipelines (inspired by [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)) that abstracts away inference details and making customization simple.

- Built on these components, dLLM provide the minimal **pretraining / finetuning / evaluation** recipes for open-weight models (e.g., [LLaDA](https://arxiv.org/abs/2502.09992) and [Dream](https://arxiv.org/abs/2508.15487)), and implementations of training algorithms (e.g., [Edit Flows](https://arxiv.org/abs/2506.09018)).

<!-- > [!NOTE]
> This repository is primarily for educational purposes and does not aim for 100% exact reproduction of official models (which is impossible). We hope it serves as a helpful reference for the community — contributions and improvements are always welcome! -->


## News
**[2025/11]** We released a collection of BERTs finetuned for instruction-following: [`ModernBERT-{large,base}-chat-v0`](https://huggingface.co/collections/dllm-collection/bert-chat). This proof-of-concept shows that BERT’s internal knowledge can be leveraged for generative tasks via masked instruction tuning. See [![blog](https://img.shields.io/badge/W&B-white?logo=weightsandbiases) BERT Chat Report](https://wandb.ai/asap-zzhou/dllm/reports/dLLM-BERT-Chat--VmlldzoxNDg0MzExNg) for detailed recipes, experimental results and lessons learned; See [`examples/bert`](/examples/bert) for training / inference / evaluation instructions.


## Table of Contents
- [Features](#features)
- [Setup](#setup)
- [Files overview](#files-overview)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Citation](#citation)


## Features
<!-- - [`examples/rnd`](/examples/rnd): (WIP) Finetuning open-weight RND1 [RND1-Base](https://www.radicalnumerics.ai/assets/rnd1_report.pdf). -->
- [`examples/llada`](/examples/llada): Pretraining, finetuning and evaluating LLaDA [LLaDA](https://arxiv.org/abs/2502.09992) / [LLaDA-MoE](https://arxiv.org/abs/2509.24389).
- [`examples/dream`](/examples/dream): Pretraining, finetuning and evaluating Dream [Dream](https://arxiv.org/abs/2508.15487).
- [`examples/bert`](/examples/bert): Finetuning any [BERT](https://arxiv.org/abs/1810.04805) to be lightweight Chatbots.
    <details>
    <summary>🎬 Click to show BERT Chat Demo</summary>

    <p align="center">
        <img src="/examples/bert/assets/chat.gif" alt="chat" width="80%">
    </p>
    <p align="center">
    <em>
        Chat with <a href="https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0"><code>ModernBERT-large-chat-v0</code></a>. See <a href="/examples/bert/README.md/#inference">Inference</a> for details.
    </em>
    </p>
    </details>
- [`examples/editflow`](/examples/editflow): Educational reference for training [EditFlow](https://arxiv.org/abs/2506.09018) models, demonstrating how to extend existing DLLMs (e.g., LLaDA, Dream, BERT Chat) with *edit operations*—insertion, deletion, and substitution—and how to pretrain or finetune EditFlow models from scratch on public data.

   <details>
   <summary>🎬 Click to show EditFlow Demo</summary>

   <p align="center">
     <img src="/examples/editflow/assets/all.gif" alt="EditFlow demo" width="100%">
   </p>
   <p align="center"><em>EditFlow performing insertion (blue), substitution from mask tokens (black), substitution from non-mask tokens (red), and deletion (strikethrough → removed) during generation.</em></p>

   </details>
- More upcoming.


## Setup
### Installation
```bash
# create and activate conda environment
conda create -n dllm python=3.10 -y
conda activate dllm

# install pytorch with CUDA 12.4 (other pytorch/cuda versions should also work)
conda install cuda=12.4 -c nvidia
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# install dllm package
pip install -e .
```
### (optional) Evaluation setup

```bash
# initialize `lm-evaluation-harness` submodule
git submodule update --init --recursive

# install submodule in editable mode with IFEval & Math dependencies
pip install -e "lm-evaluation-harness[ifeval,math]"
```

### (optional) Slurm setup
For [Slurm](https://slurm.schedmd.com/) users, update [`scripts/train.slurm.sh`](/scripts/train.slurm.sh) for your cluster:
```diff
- #SBATCH --partition=mllm_safety # Note: adjust this for your cluster
- #SBATCH --quotatype=spot        # Note: adjust this for your cluster
+ #SBATCH --partition=YOUR_PARTITION
+ #SBATCH --quotatype=YOUR_QUOTATYPE
```
Next, create a directory for your job logs:
```shell
mkdir logs
```
This folder will store the log files generated by your sbatch jobs.

## Files overview
```
# modules for training / sampling
dllm
├── core                   # Core reusable modules shared across `dllm/pipelines` 
│   ├── generation
│   ├── schedulers
│   └── trainers
├── data
├── pipelines              # Application-specific training & inference pipelines
|   ├── bert
│   ├── dream
│   ├── editflow
│   └── llada
│       ├── models         # Model architecture and configs 
│       ├── generator.py   # Generation utilities
│       ├── trainer.py     # Core training logic
│       └── eval.py        # Evaluation entry point
├── tools
└── utils

# entry points for training / sampling
examples
├── bert
├── dream
├── editflow
└── llada
    ├── chat.py            # Interactive inference example
    ├── generate.py        # Inference example
    ├── pt.py              # Pretraining example
    ├── README.md          # Documentation (you are here)
    ├── sft.py             # Supervised finetuning example
    └── eval.sh            # Evalution script
```

## Training

A typical training entry script looks like (for example, [`examples/llada/sft.py`](/examples/llada/sft.py)) looks like this:
```python
import transformers

import dllm

model_args, data_args, training_args = parser.parse_args_into_dataclasses()
# ----- Model ------------------------------------------------------------------
model = dllm.utils.get_model(model_args=model_args)
# ----- Tokenizer --------------------------------------------------------------
tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
# ----- Dataset ----------------------------------------------------------------
dataset = "..."

# ----- Training --------------------------------------------------------------
trainer = dllm.core.trainers.MDLMTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer,
        return_tensors="pt",
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id, 
    ),
)
trainer.train()
```

You can launch training job locally with `accelerate`, or submit it to a [Slurm](https://slurm.schedmd.com/) cluster using `sbatch`.
```shell
# Run locally (ZeRO-2 on 8 GPUs with 4bit quantization and LoRA)
accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/llada/sft.py \
    --num_train_epochs 4 \
    --load_in_4bit True --lora True
```
```shell
# Submit to a Slurm cluster (FSDP on 1 node, 8 GPUs)
sbatch --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/llada/sft.py" \
    --num_train_epochs 4

# Submit to a Slurm cluster (FSDP on 2 nodes, 16 GPUs)
sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/llada/sft.py" \
    --num_train_epochs 4
```
See [Features](#features) for specific training recipes.


> Here are some useful tips for training:
> 1. Use a subset of data:
> `--dataset_args "allenai/tulu-3-sft-mixture[train:10000,test:1000]"`
> 2. Concatenate datasets:
> `--dataset_args "allenai/tulu-3-sft-mixture|HuggingFaceTB/smoltalk"`
> 3. Train with LoRA and 4bit quantization:
> `--load_in_4bit True --lora True`
> 4. Train with different distributed training methods:
> `--accelerate_config "ddp,zero-{1,2,3},fsdp"`

## Inference

We provide unified [generators](/dllm/core/generation/generator.py) that abstracts away inference details. 
A typical inference entry script (for example, [`examples/llada/generate.py`](/examples/llada/generate.py)) looks like this:
```python
import dllm
from dllm import llada

model = dllm.utils.get_model(model_args=script_args).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
# for other models, change your generator and keep others unchanged
generator = llada.LLaDAGenerator(model=model, tokenizer=tokenizer)

messages = [
    [{"role": "user", "content": "Lily runs 12 km/h for 4 hours. How far in 8 hours?"}],
    [{"role": "user", "content": "Please write an educational python function."}],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)

outputs = generator.generate(inputs, return_dict_in_generate=True)
sequences = decode_trim(tokenizer, outputs.sequences.tolist(), inputs)
```

You can also try interactive chat script (for example, [`examples/llada/chat.py`](/examples/llada/chat.py)) for visualized multi-turn dialogue:

<p align="center">
    <img src="/assets/chat.gif" alt="chat" width="80%">
</p>
<!-- <p align="center"><em>EditFlow performing insertion (blue), substitution from mask tokens (black), substitution from non-mask tokens (red), and deletion (strikethrough → removed) during generation.</em></p> -->

## Evaluation
> Read [(optional) Evaluation setup](/README.md/#optional-evaluation-setup) before running evaluation. 

For example, to evaluate [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) on [`MMLU_Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), run:
```shell
accelerate launch --num_processes 4 \
    dllm/pipelines/llada/eval.py \
    --tasks "mmlu_pro" \
    --model "llada" \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,is_check_greedy=False,mc_num=1,max_new_tokens=256,steps=256,block_length=256,cfg=0.0"
```

We also provide scripts to automatically evaluate [LLaDA](https://arxiv.org/abs/2502.09992), [Dream](https://arxiv.org/abs/2508.15487), and [BERT-Chat](https://huggingface.co/collections/dllm-collection/bert-chat) on all benchmarks.
For example, you can launch [`examples/llada/eval.sh`](/examples/llada/eval.sh) directly using the following commands:
```shell
bash examples/llada/eval.sh --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" --instruct True
bash examples/llada/eval.sh --model_name_or_path "GSAI-ML/LLaDA-8B-Base" --instruct False
```


## Citation
```
@misc{dllm,
    author = {Zhanhui Zhou and Lingjie Chen and Hanghang Tong and Dawn Song},
    title = {dLLM: Simple Diffusion Language Modeling},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/ZHZisZZ/dllm}},
}
```
