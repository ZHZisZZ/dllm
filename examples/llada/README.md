# LLaDA

> 📄 Paper: [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992) | 💻 Code: [github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)

Resources and examples for training (finetuning & pretraining) and evaluating diffusion language models **LLaDA**.

## Table of Contents
- [Files](#files)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)

<!-- ## Setup
> [!IMPORTANT]  
> **Slurm users:** Update `scripts/train.slurm.sh` and `mkdir .logs`: see [(optional) Slurm setup](/README.md#optional-slurm-setup) for details.
>
> **MoE checkpoints:** For models like [`LLaDA-MoE-7B-A1B-Base`](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base), set `"model_type"` to `"lladamoe"` in the checkpoint’s `config.json`:
> ```diff
> - "model_type": "llada",
> + "model_type": "lladamoe",
> ```
> -->


## Files
```
# Pipeline modules relevant to LLaDA
dllm/pipelines/llada
├── __init__.py                     # Package initialization
├── models/
│   ├── configuration_lladamoe.py   # LLaDA-MoE model configuration
│   ├── configuration_llada.py      # LLaDA model configuration
│   ├── modeling_lladamoe.py        # LLaDA-MoE model architecture
│   └── modeling_llada.py           # LLaDA model architecture
├── eval.py                         # Evaluation module
├── sampler.py                      # Inference module
└── trainer.py                      # Training module (pretraining, SFT, and GRPO/RL)

# Example entry points for training / inference / evaluation
examples/llada
├── chat.py                         # Interactive inference example
├── eval.sh                         # Automatic evaluation example
├── grpo.py                         # GRPO/RL training entry point
├── sample.py                       # Inference example
├── pt.py                           # Pretraining example
├── README.md                       # Documentation (you are here)
└── sft.py                          # Supervised finetuning example
```
<!-- > [!NOTE] -->
<!-- >  - We fixed attention mask bugs in [`modeling_lladamoe.py`](/dllm/pipelines/llada/models/modeling_lladamoe.py) and [`modeling_llada.py`](/dllm/pipelines/llada/models/modeling_llada.py). We recommend loading models with `dllm.utils.get_tokenizer`; otherwise `import dllm` before calling `AutoModel.from_pretrained` to ensure the correct models from `dllm` are used. 
> 
>  - We fixed bugs in `chat_template` and assign `mask_token` through `dllm.utils.get_tokenizer`. If you use `AutoTokenizer`, keep in mind to set `chat_template` and `mask_token` appropriately yourselves. -->

<!-- > [!WARNING]  
> Before loading MoE checkpoints (e.g., [inclusionAI/LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base)), first overwrite the `model_type` field from `inclusionAI/LLaDA-MoE-7B-A1B-Base/config.json`:  
> ```diff
> - "model_type": "llada",
> + "model_type": "lladamoe",
> ``` -->

## Training

> Read [Useful tips for training](/README.md#useful-tips-for-training) and [(optional) Slurm setup](/README.md#optional-slurm-setup) before training.
>
> **MoE checkpoints:** For models like [`LLaDA-MoE-7B-A1B-Base`](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base), set `"model_type"` to `"lladamoe"` in the checkpoint’s `config.json`:
<!-- > ```diff
> - "model_type": "llada",
> + "model_type": "lladamoe",
> ```
> -->

### SFT

For example, to SFT [`LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) on the [`alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset for instruction following on 8 GPUs, run:
```shell
accelerate launch \
    --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/llada/sft.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 1024 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir ".models/LLaDA-8B-Base/alpaca"
```
If you are using slurm and want to train across, for example, 2 nodes (16 GPUs total), run:
```shell
sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/llada/sft.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 1024 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir ".models/LLaDA-8B-Base/alpaca"
```

<!-- **Reproducing [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)**. Though LLaDA is trained on proprietary data, we tried our best to reproduce LLaDA-8B-Instruct by finetuning LLaDA-8B-Base using our training pipeline on public instruction-following dataset [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture): -->

#### Reproducing [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) with SFT
Though LLaDA is trained on proprietary data, we tried our best to reproduce [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) by finetuning [`LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) with SFT on the [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) dataset:

```shell
# Preprocessing SFT data (optional, but can avoid redundant preprocessing for multi-node training)
python dllm/tools/preprocess_sft_dataset.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --sft_map_fn_path "dllm.utils.default_sft_map_fn" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir ".data/sft/llada/tulu-3-sft-mixture" \
    --num_proc 64

# Train on 24*8=192 A100s with FSDP, take about 8 hours
sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/llada/sft.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args ".data/sft/llada/tulu-3-sft-mixture" \
    --load_preprocessed_data True \
    --max_length 1024 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir ".models/LLaDA-8B-Base/tulu-3-sft-mixture"
```
<!-- [TODO] Training curves are on Wandb; checkpoints with evaluation results are available on Hugging Face. See the [Evaluation](#evaluation) section below for evaluation instructions. -->


### RL (GRPO)

We adapt [GRPO](https://arxiv.org/abs/2402.03300) (Group Relative Policy Optimization) for masked diffusion language models via `DiffuGRPOTrainer`, which replaces autoregressive generation with iterative denoising. The implementation follows the [d1/diffu-grpo](https://github.com/dllm-reasoning/d1/tree/main/diffu-grpo) reference.

For example, to run GRPO on [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) with `gsm8k` on 1 GPU:
```shell
accelerate launch \
    --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/llada/grpo.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --dataset gsm8k \  # supported: gsm8k, countdown, sudoku, math, code
    --num_train_epochs 1 \
    --output_dir ".models/LLaDA-8B-Instruct/gsm8k-grpo"
```

To train with LoRA on 8 GPUs using DeepSpeed ZeRO-2:
```shell
accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/llada/grpo.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --lora_r 128 --lora_alpha 64 \
    --dataset gsm8k \
    --num_train_epochs 10 --learning_rate 3e-6 \
    --num_generations 6 --per_device_train_batch_size 6 \
    --beta 0.04 --epsilon 0.5 \
    --output_dir ".models/LLaDA-8B-Instruct/gsm8k-grpo"
```

Key diffusion-specific arguments: `--block_size`, `--steps`, `--remasking`, `--p_mask_prompt`.
Key GRPO arguments: `--beta`, `--epsilon`, `--num_generations`, `--num_iterations`.

### Pretraining

Pretrain on [`mlfoundations/dclm-baseline-1.0`](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) from scratch using 192 GPUs (24x8) and FSDP:
```shell
sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/llada/pt.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "mlfoundations/dclm-baseline-1.0" \
    --max_length 1024 \
    --max_steps 2000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir ".models/LLaDA-8B-Base/dclm-baseline-1.0"
```

## Inference
We support batch inference for standard sampling and infilling:
```shell
python examples/llada/sample.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct"
```
We also support interactive multi-turn dialogue with visualization:
```shell
python examples/llada/chat.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct"
```

## Evaluation
> Read [(optional) Evaluation setup](/README.md#optional-evaluation-setup) before running evaluation. 

For example, to evaluate [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) on [gsm8k](https://huggingface.co/datasets/openai/gsm8k) using 4 GPUs, run:
```shell
# Use model_args to adjust the sampling arguments for evaluation.
accelerate launch --num_processes 4 \
    dllm/pipelines/llada/eval.py \
    --tasks "gsm8k_cot" \
    --model "llada" \
    --apply_chat_template \
    --num_fewshot 5 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,max_new_tokens=512,steps=512,block_size=512,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"
```

To automatically evaluate [`LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) on all benchmarks, run:
```shell
bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Instruct --instruct True
bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Base --instruct False
```

For **Fast-dLLM** sampling and evaluation with LLaDA, see the [Fast-dLLM README](../fastdllm/README.md).

For **Info-Gain** sampling and evaluation with LLaDA, see the [Info-Gain README](../infogain/README.md).

### Evaluation results

> Results (Reproduced) are evaluated using our framework, while results (Official) come from the original [paper](https://arxiv.org/abs/2502.09992). All evaluation settings follow the configurations in the [LLaDA](https://github.com/ML-GSAI/LLaDA) repository, with minor adjustments. 

|               | MMLU | BBH | ARC&#8209;C | Hellaswag | TruthfulQA | WinoGrande | PIQA | GSM8K | Math | GPQA | HumanEval | MBPP | CEval | CMMLU |
|:----------------|:----:|:-----:|:-----------:|:-----------:|:------------:|:----:|:-----:|:----:|:-----:|:----:|:-----------:|:----:|:------:|:------:|
| [`LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) (Official) | 65.9 | 49.7 | 45.9 | 70.5 | 46.1 | 74.8 | 73.6 | 70.3 | 31.4 | 25.2 | 35.4 | 40.0 | 70.5 | 69.9 |
| [`LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) (Reproduced) | 65.9 | 47.2 | 44.1 | 69.2 | 45.6 | 70.4 | 70.7 | 70.7 | 32.4 | 31.9 | 32.9 | 38.8 | 70.4 | 69.8 |


<p align="center" style="color: #808080; font-size: 0.9em;">
Table 1. Evaluation results of 
<a href="https://huggingface.co/GSAI-ML/LLaDA-8B-Base" style="color: #808080; text-decoration: none;">
<code>LLaDA-8B-Base</code>
</a>.
</p>

|                 | MMLU | MMLU&#8209;Pro | ARC&#8209;C | Hellaswag | GSM8K | Math | GPQA | HumanEval | MBPP | 
|:----------------|:----:|:---------:|:-----:|:-----------:|:-----:|:----:|:----:|:-----------:|:----:|
| [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) (Official) | 65.5 | 37.0 | 88.5 | 74.6 | 69.4 | 31.9 | 33.3 | 49.4 | 41.0 |
| [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) (Reproduced) | 69.8 | 36.2 | 86.4 | 76.7 | 74.7 | 31.9 | 30.6 | 47.0 | 40.0 |

<p align="center" style="color: #808080; font-size: 0.9em;">
Table 2. Evaluation results of 
<a href="https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct" style="color: #808080; text-decoration: none;">
<code>LLaDA-8B-Instruct</code>
</a>.
</p>
