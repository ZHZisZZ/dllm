# BERT

[![Hugging Face Checkpoints](https://img.shields.io/badge/Hugging%20Face-Checkpoints-yellow)](https://huggingface.co/collections/dllm-collection/bert-chat)
[![W&B Report](https://img.shields.io/badge/W&B-Report-white?logo=weightsandbiases)](https://api.wandb.ai/links/asap-zzhou/101h5xvg)

This directory provides two key sets of resources:

-  **[Warmup](#warmup)**: Tutorials for continual pretraining and SFTing any BERT-style model on small datasets to generate text with diffusion.
-  **[`BERT-Chat`](#bert-chat)**: The exact training, inference, and evaluation scripts used to develop the 🤗checkpoints: [`ModernBERT-base-chat-v0.1`](https://huggingface.co/dllm-collection/ModernBERT-base-chat-v0.1) and [`ModernBERT-large-chat-v0.1`](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0.1), two BERTs finetuned as Chatbots via SFT. For a deep dive into experimental results, lessons learned, and more reproduction details, please see our full [![blog](https://img.shields.io/badge/W&B-white?logo=weightsandbiases) BERT-Chat Report](https://api.wandb.ai/links/asap-zzhou/101h5xvg).

<p align="center" style="margin-top: 15px;">
    <img src="/examples/bert/assets/chat.gif" alt="chat" width="70%">
</p>
<p align="center">
  <em>
    Chat with <a href="https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0.1"><code>ModernBERT-large-chat-v0.1</code></a>. See <a href="/examples/bert/README.md/#inference">Inference</a> for details.
  </em>
</p>

## Files overview
```
# example entry points for training / inference / evaluation
examples/bert
├── chat.py                         # Interactive inference example
├── eval.sh                         # Automatic evaluation example
├── sample.py                       # Inference example
├── pt.py                           # Pretraining example
├── README.md                       # Documentation (you are here)
└── sft.py                          # Supervised finetuning example
```

## Warmup

In this section, we show toy examples of continual pretraining and SFTing [`ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) on small datasets to generate text.
You can use any BERT model instead for example, by `--model_name_or_path "FacebookAI/roberta-large"`.

### Continual Pretraining

To train [`ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) on the [`tiny-shakespeare`](https://huggingface.co/datasets/Trelis/tiny-shakespeare) dataset, run:
```shell
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/bert/pt.py \
    --model_name_or_path "answerdotai/ModernBERT-large" \
    --dataset_args "Trelis/tiny-shakespeare" \
    --text_field "Text" \
    --insert_eos False \
    --max_length 128 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --eval_steps 0.1 \
    --save_steps 0.1 \
    --output_dir "models/ModernBERT-large/tiny-shakespeare"
```

To sample from the model interactively:
```shell
# Enter a prompt (e.g., "First citizen: Before we proceed any further, hear me speak."),
# or press Enter to let the model generate text from scratch.
python -u examples/bert/chat.py \
    --model_name_or_path "models/ModernBERT-large/tiny-shakespeare/checkpoint-final" \
    --chat_template False --remasking "random" --steps 128 --max_new_tokens 128
```

### SFT

To train [`ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) on the [`alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset, run:
```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/bert/sft.py \
    --model_name_or_path "answerdotai/ModernBERT-large" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 512 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --eval_steps 0.1 \
    --save_steps 0.1 \
    --output_dir "models/ModernBERT-large/alpaca"
```

To chat with the model:
```shell
python -u examples/bert/chat.py \
    --model_name_or_path "models/ModernBERT-large/alpaca/checkpoint-final"
```

## `BERT-Chat`
Here we show the exact commands we use to train / interact with / evaluation the [`BERT-Chat`](https://huggingface.co/collections/dllm-collection/bert-chat) models: 
[`ModernBERT-base-chat-v0.1`](https://huggingface.co/dllm-collection/ModernBERT-base-chat-v0.1) and [`ModernBERT-large-chat-v0.1`](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0.1).
For training curves and other details, please see [![blog](https://img.shields.io/badge/W&B-white?logo=weightsandbiases) BERT-Chat Report](https://api.wandb.ai/links/asap-zzhou/101h5xvg).

### Training
> Read [Useful tips for training](/README.md/#useful-tips-for-training) before training.

The [`BERT-Chat`](https://huggingface.co/collections/dllm-collection/bert-chat) models are trained purely with SFT on the [`tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) and [`smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) dataset.

To reproduce [`ModernBERT-base-chat-v0.1`](https://huggingface.co/dllm-collection/ModernBERT-base-chat-v0.1), run the command below (about 4 hours on 8 GPUs):
```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/bert/sft.py \
    --model_name_or_path "answerdotai/ModernBERT-base" \
    --dataset_args "allenai/tulu-3-sft-mixture+HuggingFaceTB/smoltalk" \
    --max_length 1024 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 48 \
    --eval_steps 0.1 \
    --save_steps 0.1 \
    --output_dir "models/ModernBERT-base/tulu-3-sft-mixture+smoltalk"
```

To reproduce [`ModernBERT-large-chat-v0.1`](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0.1), run the command below (about 7 hours on 8 GPUs):
```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/bert/sft.py \
    --model_name_or_path "answerdotai/ModernBERT-large" \
    --dataset_args "allenai/tulu-3-sft-mixture+HuggingFaceTB/smoltalk" \
    --max_length 1024 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 48 \
    --eval_steps 0.1 \
    --save_steps 0.1 \
    --output_dir "models/ModernBERT-large/tulu-3-sft-mixture+smoltalk"
```

### Inference

To chat with the model:
```shell
python -u examples/bert/chat.py --model_name_or_path "dllm-collection/ModernBERT-large-chat-v0.1"
```

## Evaluation
> Read [(optional) Evaluation setup](/README.md/#optional-evaluation-setup) before running evaluation.

For example, to evaluate [`ModernBERT-large-chat-v0.1`](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0.1) on [`gsm8k`](https://huggingface.co/datasets/openai/gsm8k) using 4 GPUs, run:
```shell
# Use model_args to adjust the sampler arguments for evaluation.
accelerate launch --num_processes 4 \
    dllm/pipelines/bert/eval.py \
    --tasks "gsm8k_bert" \
    --model "bert" \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=dllm-collection/ModernBERT-large-chat-v0.1,max_new_tokens=256,steps=256,block_size=32"
```

To automatically evaluate [`ModernBERT-base-chat-v0.1`](https://huggingface.co/dllm-collection/ModernBERT-base-chat-v0.1) and [`ModernBERT-large-chat-v0.1`](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0.1) on all benchmarks, run:
```shell
bash examples/bert/eval.sh --model_name_or_path "dllm-collection/ModernBERT-base-chat-v0.1"
bash examples/bert/eval.sh --model_name_or_path "dllm-collection/ModernBERT-large-chat-v0.1"
```

### Evaluation Results
<!-- 
|                     | LAMBADA | GSM8K | CEval | BBH | MATH | MMLU | Winogrande | HellaSwag | CMMLU |
|:------------------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| [`ModernBERT-base-chat-v0.1`](https://huggingface.co/dllm-collection/ModernBERT-base-chat-v0.1)(evaluated) | 49.3 | 5.9 | 25.0 | 17.9 | 3.1 | 26.1 | 49.7 | 41.0 | 24.3 |
| [`ModernBERT-large-chat-v0.1`](https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0.1)(evaluated) | 46.3 | 17.1 | 24.6 | 25.1 | 3.8 | 33.5 | 53.1 | 45.0 | 27.5 |
| [`Qwen1.5-0.5B`](https://huggingface.co/Qwen/Qwen1.5-0.5B)(<ins>reported</ins> & evaluated) | 48.6 | <ins>22.0</ins> | <ins>50.5</ins> | <ins>18.3</ins> | <ins>3.1</ins> | <ins>39.2</ins> | 55.0 | 48.2 | <ins>46.6</ins> |
| [`Qwen1.5-0.5B-Chat`](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat)(<ins>reported</ins> & evaluated) | 41.2 | <ins>11.3</ins> | <ins>37.2</ins> | 18.2 | 2.1 | <ins>35.0</ins> | 52.0 | 36.9 | 32.2 |
| [`gpt2`](https://huggingface.co/openai-community/gpt2)(<ins>reported</ins> & evaluated) | <ins>46.0</ins> | 0.7 | 24.7 | 6.9 | 1.8 | 22.9 | 51.6 | 31.1  | 25.2 |
| [`gpt2-medium`](https://huggingface.co/openai-community/gpt2-medium)(<ins>reported</ins> & evaluated) | <ins>55.5</ins> | 2.1 | 24.6 | 17.8 | 1.4 | 22.9 | 53.1 | 39.4 | 0.3 | -->

<table style="border-collapse: collapse; width: 100%; text-align: center; table-layout: fixed;">
  <colgroup>
    <col style="width: 28%;">  <!-- widen first column -->
    <col style="width: 8%;">
    <col style="width: 8%;">
    <col style="width: 8%;">
    <col style="width: 8%;">
    <col style="width: 8%;">
    <col style="width: 8%;">
    <col style="width: 8%;">
    <col style="width: 8%;">
  </colgroup>

  <thead>
    <tr style="border-bottom: 3px solid #333;">
      <th style="padding: 8px; text-align: left;">
        Model
      </th>
      <th style="padding: 8px;">LAMBADA</th>
      <th style="padding: 8px;">GSM8K</th>
      <th style="padding: 8px;">CEval</th>
      <th style="padding: 8px;">BBH</th>
      <th style="padding: 8px;">MATH</th>
      <th style="padding: 8px;">MMLU</th>
      <th style="padding: 8px;">Winogrande</th>
      <th style="padding: 8px;">HellaSwag</th>
      <th style="padding: 8px;">CMMLU</th>
    </tr>
  </thead>

  <!-- ModernBERT-base -->
  <tr>
    <td style="padding: 8px;">
      <a href="https://huggingface.co/dllm-collection/ModernBERT-base-chat-v0.1"><code>ModernBERT-base-chat-v0.1</code></a>(evaluated)
    </td>
    <td>49.3</td><td>5.9</td><td>25.0</td><td>17.9</td><td>3.1</td><td>26.1</td><td>49.7</td><td>41.0</td><td>24.3</td>
  </tr>

  <!-- ModernBERT-large -->
  <tr>
    <td style="padding: 8px;">
      <a href="https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0.1"><code>ModernBERT-large-chat-v0.1</code></a>(evaluated)
    </td>
    <td>46.3</td><td>17.1</td><td>24.6</td><td>25.1</td><td>3.8</td><td>33.5</td><td>53.1</td><td>45.0</td><td>27.5</td>
  </tr>

  <tr>
    <td colspan="10" style="padding: 0; border-top: 3px double #666;"></td>
  </tr>

<!-- Qwen1.5-0.5B -->
<tr>
  <td style="padding: 8px;"><i>
    <a href="https://huggingface.co/Qwen/Qwen1.5-0.5B"><code>Qwen1.5-0.5B</code></a>(<ins>reported</ins> & evaluated)
  </i></td>
  <td><i>48.6</i></td><td><i><ins>22.0</ins></i></td><td><i><ins>50.5</ins></i></td><td><i><ins>18.3</ins></i></td><td><i><ins>3.1</ins></i></td><td><i><ins>39.2</ins></i></td><td><i>55.0</i></td><td><i>48.2</i></td><td><i><ins>46.6</ins></i></td>
</tr>

<!-- Qwen1.5-0.5B-Chat -->
<tr>
  <td style="padding: 8px;"><i>
    <a href="https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat"><code>Qwen1.5-0.5B-Chat</code></a>(<ins>reported</ins> & evaluated)
  </i></td>
  <td><i>41.2</i></td><td><i><ins>11.3</ins></i></td><td><i><ins>37.2</ins></i></td><td><i>18.2</i></td><td><i>2.1</i></td><td><i><ins>35.0</ins></i></td><td><i>52.0</i></td><td><i>36.9</i></td><td><i>32.2</i></td>
</tr>

<!-- gpt2 -->
<tr>
  <td style="padding: 8px;"><i>
    <a href="https://huggingface.co/openai-community/gpt2"><code>gpt2</code></a>(<ins>reported</ins> & evaluated)
  </i></td>
  <td><i><ins>46.0</ins></i></td><td><i>0.7</i></td><td><i>24.7</i></td><td><i>6.9</i></td><td><i>1.8</i></td><td><i>22.9</i></td><td><i>51.6</i></td><td><i>31.1</i></td><td><i>25.2</i></td>
</tr>

<!-- gpt2-medium -->
<tr>
  <td style="padding: 8px;"><i>
    <a href="https://huggingface.co/openai-community/gpt2-medium"><code>gpt2-medium</code></a>(<ins>reported</ins> & evaluated)
  </i></td>
  <td><i><ins>55.5</ins></i></td><td><i>2.1</i></td><td><i>24.6</i></td><td><i>17.8</i></td><td><i>1.4</i></td><td><i>22.9</i></td><td><i>53.1</i></td><td><i>39.4</i></td><td><i>0.3</i></td>
</tr>


</table>


<p align="left" style="color: #808080; font-size: 0.9em;">
Table 1. Evaluation results of 
<a href="https://huggingface.co/dllm-collection/ModernBERT-base-chat-v0.1" style="color: #808080; text-decoration: none;">
<code>ModernBERT-base-chat-v0.1</code>
</a>,
<a href="https://huggingface.co/dllm-collection/ModernBERT-large-chat-v0.1" style="color: #808080; text-decoration: none;">
<code>ModernBERT-large-chat-v0.1</code>
</a>,
<a href="https://huggingface.co/Qwen/Qwen1.5-0.5B" style="color: #808080; text-decoration: none;">
<code>Qwen1.5-0.5B</code>
</a>,
<a href="https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat" style="color: #808080; text-decoration: none;">
<code>Qwen1.5-0.5B-Chat</code>
</a>,
<a href="https://huggingface.co/openai-community/gpt2" style="color: #808080; text-decoration: none;">
<code>gpt2</code>
</a>, and
<a href="https://huggingface.co/openai-community/gpt2-medium" style="color: #808080; text-decoration: none;">
<code>gpt2-medium</code>
</a>.
<ins>Underlined entries</ins> are results from official reports: <a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" style="color: #808080; text-decoration: none;">GPT-2 paper</a>, <a href="https://qwen.ai/blog?id=qwen1.5" style="color: #808080; text-decoration: none;">Qwen1.5 blog</a>, and <a href="https://huggingface.co/Qwen/Qwen2-0.5B-Instruct" style="color: #808080; text-decoration: none;">Qwen2-0.5B-Instruct model card</a>. All other results are evaluated using our framework. <i>Italic rows</i> denote autoregressive models, whereas non-italic rows denote diffusion language models.
</p>
