# Generative BERT

[![Hugging Face Checkpoints](https://img.shields.io/badge/Hugging%20Face-Checkpoints-yellow)](https://huggingface.co/collections/dllm-collection/bert-chat)
[![W&B Report](https://img.shields.io/badge/W&B-Report-white?logo=weightsandbiases)](https://api.wandb.ai/links/asap-zzhou/101h5xvg)

This directory provides two key sets of resources:

1.  **Toy Examples ([Warmup](#warmup)):** Scripts for pretraining and SFTing any BERT-style model on small datasets to generate text.
2.  **Official Scripts ([BERT Chat](#bert-chat)):** The exact training, inference, and evaluation scripts used to create the [`ModernBERT-base-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-base-chat-v0) and [`ModernBERT-large-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0) checkpoints, two BERTs finetuned as Chatbots. For a deep dive into experimental results, lessons learned, and more reproduction details, please see our full [BERT Chat W&B Report](https://api.wandb.ai/links/asap-zzhou/101h5xvg).

<p align="center" style="margin-top: 15px;">
    <img src="/examples/bert/assets/chat.gif" alt="chat" width="70%">
</p>
<p align="center">
  <em>
    Chat with <a href="https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0"><code>ModernBERT-large-chat-v0</code></a>. See <a href="/examples/bert/README.md/#inference">Inference</a> for details.
  </em>
</p>

## Files overview
```
# example entry points for training / inference / evaluation
examples/bert
├── chat.py                         # Interactive inference example
├── eval.sh                         # Automatic evaluation script
├── generate.py                     # Inference example
├── pt.py                           # Pretraining example
├── README.md                       # Documentation (you are here)
└── sft.py                          # Supervised finetuning example
```

## Warmup

In this section, we show toy examples of pretraining and SFTing [`ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) on small datasets to generate text.
You can use any BERT model instead for example, by `--model_name_or_path "FacebookAI/roberta-large"`.

### Pretrain

To train [`ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) on the [`tiny-shakespeare`](https://huggingface.co/datasets/Trelis/tiny-shakespeare) dataset, run:
```shell
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/bert/pt.py \
    --model_name_or_path "answerdotai/ModernBERT-large" \
    --dataset_args "Trelis/tiny-shakespeare" \
    --text_field "Text" \
    --insert_eos False \
    --max_length 128 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --save_steps 0.1 \
    --output_dir "models/ModernBERT-large/tiny-shakespeare"
```

To run inference with the model:
```shell
# just press enter (empty prompt) if you want the model to generate text from scratch 
python -u examples/bert/chat.py \
    --model_name_or_path "models/ModernBERT-large/tiny-shakespeare/checkpoint-final" \
    --chat False --remasking "random" --steps 128 --max_new_tokens 128
```

### SFT

To train [`ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) on the [`alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset, run:
```shell
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 8 \
    examples/bert/sft.py \
    --model_name_or_path "answerdotai/ModernBERT-large" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 512 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --save_steps 0.1 \
    --output_dir "models/ModernBERT-large/alpaca"
```

To chat with the model:
```shell
python -u examples/bert/chat.py \
    --model_name_or_path "models/ModernBERT-large/alpaca/checkpoint-final" --chat True
```

## BERT Chat
Here we show the exact commands we use to train and interact with the BERT Chat models: 
[`ModernBERT-base-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-base-chat-v0) and [`ModernBERT-large-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0).
For training curves and other details, please see [BERT Chat W&B Report](https://api.wandb.ai/links/asap-zzhou/101h5xvg).

### Training

To reproduce [`ModernBERT-base-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-base-chat-v0), run:
```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/bert/sft.py \
    --model_name_or_path "answerdotai/ModernBERT-base" \
    --dataset_args "allenai/tulu-3-sft-mixture|HuggingFaceTB/smoltalk" \
    --max_length 1024 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 48 \
    --save_steps 0.1 \
    --output_dir "models/ModernBERT-base/tulu-3-smoltalk/epochs-10-bs-384-len-1024"
```

To reproduce [`ModernBERT-large-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0), run:
```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/bert/sft.py \
    --model_name_or_path "answerdotai/ModernBERT-large" \
    --dataset_args "allenai/tulu-3-sft-mixture|HuggingFaceTB/smoltalk" \
    --max_length 1024 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 48 \
    --save_steps 0.1 \
    --output_dir "models/ModernBERT-large/tulu-3-smoltalk/epochs-10-bs-384-len-1024"
```

### Inference

To chat with the model:
```shell
python -u examples/bert/chat.py --model_name_or_path "dllm-collection/ModernBERT-large-chat-v0" --chat True
```

## Evaluation
> Read [(optional) Evaluation setup](/README.md/#optional-evaluation-setup) before running evaluation. 

For example, to evaluate [`ModernBERT-large-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0) on [`MMLU-Pro`](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) using 4 GPUs, run:
```shell
# Use model_args to adjust the generation arguments for evalution.
accelerate launch  --num_processes 4 \
    dllm/pipelines/bert/eval.py \
    --tasks "mmlu_pro" \
    --model "bert" \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=dllm-collection/ModernBERT-large-chat-v0,is_check_greedy=False,mc_num=1,max_new_tokens=256,steps=256,block_length=256"
```

To automatically evaluate [`ModernBERT-base-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-base-chat-v0) and [`ModernBERT-large-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0) on all benchmarks, run:
```shell
bash examples/bert/eval.sh --model_name_or_path "dllm-collection/ModernBERT-base-chat-v0"
bash examples/bert/eval.sh --model_name_or_path "dllm-collection/ModernBERT-large-chat-v0"
```

### Evaluation results

<!-- > Evaluated results are obtained using our own evaluation framework, while Reported results are taken from the original paper. 
> Because the original work does not fully disclose its evaluation techniques or implementation tricks, we reproduce the setup using the best available methods. As a result, our reproduced scores may show a small residual gap relative to the reported numbers.  -->

<!-- | [`GPT-2`](https://huggingface.co/openai-community/gpt2)(reported) | 0.460 | – |  |  |  |  |  |  |  |
| [`GPT-2`](https://huggingface.co/openai-community/gpt2)(evaluated) | 0.438 | 0.020 |  |  |  |  |  |  |  |
| [`GPT-2-medium`](https://huggingface.co/openai-community/gpt2-medium)(reported) | 0.555 | – |  |  |  |  |  |  |  |
| [`GPT-2-medium`](https://huggingface.co/openai-community/gpt2-medium)(evaluated) | 0.549 | 0.021 |  |  |  |  |  |  |  | -->
<!-- <div align="center" style="min-width:1500px;"> -->

|                     | LAMBADA | GSM8K | CEval | BBH | MATH | MMLU | Winogrande | HellaSwag | CMMLU |
|:------------------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| [`ModernBERT-base-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-base-chat-v0)(evaluated) | 49.3 | 5.9 | 25.0 | 17.9 | 3.1 | 26.1 | 49.7 | 41.0 | 24.3 |
| [`ModernBERT-large-chat-v0`](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0)(evaluated) | 46.3 | 17.1 | 24.6 | 25.1 | 3.8 | 33.5 | 53.1 | 45.0 | 27.5 |
| [`Qwen1.5-0.5B`](https://huggingface.co/Qwen/Qwen1.5-0.5B)(<ins>reported</ins> & evaluated) | 48.6 | <ins>22.0</ins> | <ins>50.5</ins> | <ins>18.3</ins> | <ins>3.1</ins> | <ins>39.2</ins> | 55.0 | 48.2 | <ins>46.6</ins> |
| [`Qwen1.5-0.5B-chat`](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat)(<ins>reported</ins> & evaluated) | 41.2 | <ins>11.3</ins> | <ins>37.2</ins> | 18.2 | 2.1 | <ins>35.0</ins> | 52.0 | 36.9 | 32.2 |
| [`GPT-2`](https://huggingface.co/openai-community/gpt2)(<ins>reported</ins> & evaluated) | <ins>46.0</ins> | 0.7 | 24.7 | 6.9 | 1.8 | 22.9 | 51.6 | 31.1  | 25.2 |
| [`GPT-2-medium`](https://huggingface.co/openai-community/gpt2-medium)(<ins>reported</ins> & evaluated) | <ins>55.5</ins> | 2.1 | 24.6 | 17.8 | 1.4 | 22.9 |53.1  | 39.4  | 0.3  |


<p align="left" style="color: #808080; font-size: 0.9em;">
Table 1. Evaluation results of 
<a href="https://huggingface.co/dllm-collection/ModernBERT-base-chat-v0" style="color: #808080; text-decoration: none;">
<code>ModernBERT-base-chat-v0</code>
</a>,
<a href="https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0" style="color: #808080; text-decoration: none;">
<code>ModernBERT-large-chat-v0</code>
</a>,
<a href="https://huggingface.co/Qwen/Qwen1.5-0.5B" style="color: #808080; text-decoration: none;">
<code>Qwen1.5-0.5B</code>
</a>,
<a href="https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat" style="color: #808080; text-decoration: none;">
<code>Qwen1.5-0.5B-Chat</code>
</a>,
<a href="https://huggingface.co/openai-community/gpt2" style="color: #808080; text-decoration: none;">
<code>GPT-2</code>
</a>, and
<a href="https://huggingface.co/openai-community/gpt2-medium" style="color: #808080; text-decoration: none;">
<code>GPT-2-medium</code>
</a>.
<ins>Underlined entries</ins> are results from official reports: <a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" style="color: #808080; text-decoration: none;">GPT-2 paper</a>, <a href="https://qwen.ai/blog?id=qwen1.5" style="color: #808080; text-decoration: none;">Qwen 1.5 blog</a>, and <a href="https://huggingface.co/Qwen/Qwen2-0.5B-Instruct" style="color: #808080; text-decoration: none;">Qwen2-0.5B-Instruct model card</a>. All other results are evaluated using our framework.
</p>
