# LLaDA

> 📄 Paper: [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992) | 💻 Code: [github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)

Resources and examples for training (finetuning & pretraining) and evaluating diffusion language models **LLaDA**.

## Table of Contents
- [Setup](#setup)
- [Files overview](#files-overview)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)

## Setup
> [!IMPORTANT]  
> **Slurm users:** Update `scripts/train.slurm.sh` and `mkdir logps`: see [(optional) Slurm setup](/README.md/#optional-slurm-setup) for details.
>
> **MoE checkpoints:** For models like [`LLaDA-MoE-7B-A1B-Base`](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base), set `"model_type"` to `"lladamoe"` in the checkpoint’s `config.json`:
> ```diff
> - "model_type": "llada",
> + "model_type": "lladamoe",
> ```
>


##  Files overview
```
# tools relevant with LLaDA
dllm/pipelines/llada
├── __init__.py                     # Package initialization
├── models/
│   ├── configuration_lladamoe.py   # LLaDA-MoE model configuration
│   ├── configuration_llada.py      # LLaDA model configuration
│   ├── modeling_lladamoe.py        # LLaDA-MoE model architecture
│   └── modeling_llada.py           # LLaDA model architecture
├── generator.py                    # Inference logic
└── trainer.py                      # Training logic (pretraining and finetuning)

# example entry points for training / inference / evaluation
examples/llada
├── chat.py                         # Interactive inference example
├── eval.sh                         # Automatic evaluation script
├── generate.py                     # Inference example
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
### Finetuning

For example, to SFT [`LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) for instruction following on 8 GPUs, run:
```shell
accelerate launch \
    --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/llada/sft.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/LLaDA-8B-SFT/tulu-3-sft-mixture" \
    --max_length 1024 \ 
    --num_train_epochs 4 \
    --learning_rate 2e-5
```
If you are using slurm and want to train across, for example, 2 nodes (16 GPUs total), run:
```shell
sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/llada/sft.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/LLaDA-8B-SFT/tulu-3-sft-mixture" \
    --max_length 1024 \ 
    --num_train_epochs 4 \
    --learning_rate 2e-5
```

<!-- **Reproducing [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)**. Though LLaDA is trained on proprietary data, we tried our best to reproduce LLaDA-8B-Instruct by finetuning LLaDA-8B-Base using our training pipeline on public instruction-following dataset [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture): -->

#### Reproducing [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
Though LLaDA is trained on proprietary data, we tried our best to reproduce [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) by finetuning [`LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) using our training pipeline on public instruction-following dataset [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture):

```shell
# preprocessing SFT data (optional, but can avoid redundant preprocessing for multi-node training)
python dllm/tools/preprocess_sft_dataset.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --sft_map_fn_path "dllm.utils.default_sft_map_fn" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "data/sft/llada/tulu-3-sft-mixture" \
    --num_proc 64

# train on 24*8=192 A100s with FSDP, take about 8 hours
sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/llada/sft.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "data/sft/llada/tulu-3-sft-mixture" \
    --load_preprocessed_data True \
    --output_dir "models/LLaDA-8B-SFT-tulu3-fsdp-bs4-len2048-ep5-lr1e-5" \
    --max_length 2048 \
    --truncation "right" \
    --group_by_length True \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --eval_on_start False \
    --eval_steps 0.1 \
    --save_steps 0.05
```
[TODO] Training curves are on Wandb; checkpoints with evaluation results are available on Hugging Face. See the [Evaluation](#evaluation) section below for evaluation instructions.


### Pretraining

Pretrain on [`mlfoundations/dclm-baseline-1.0`](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) from scratch using 192 GPUs (24x8) and FSDP:
```shell
sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/llada/pt.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "mlfoundations/dclm-baseline-1.0" \
    --output_dir "models/LLaDA-8B-PT/dclm-baseline-1.0" \
    --max_length 1024 \ 
    --max_steps 2000 \
    --learning_rate 3e-4
```

## Inference
We support batch inference for standard generation and infilling:
<!-- See [`examples/llada/generate.py`](/examples/llada/generate.py) for a full example: -->
```shell
python examples/llada/generate.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct"
```
We also support interactive multi-turn dialogue with visualization:
<!-- See [`examples/llada/chat.py`](/examples/llada/chat.py) for a full example. -->
```shell
python examples/llada/chat.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct"
```

## Evaluation
> Read [(optional) Evaluation setup](/README.md/#optional-evaluation-setup) before running evaluation. 

For example, to evaluate [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) on [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) using 4 GPUs, run:
```shell
# Use model_args to adjust the generation arguments for evalution.
accelerate launch --num_processes 4 \
    dllm/pipelines/llada/eval.py \
    --tasks "mmlu_pro" \
    --model "llada" \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,is_check_greedy=False,mc_num=1,max_new_tokens=256,steps=256,block_length=256,cfg=0.0"
```

To automatically evaluate [`LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) on all benchmarks, run:
```shell
bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Instruct --instruct True
bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Base --instruct False
```

### Evaluation results

<!-- > Evaluated results are obtained using our own evaluation framework, while Reported results refer to those from the original paper.  
> All evaluation parameters follow the configurations in the [LLaDA](https://github.com/ML-GSAI/LLaDA) repository.  
> Since the original evaluations were conducted with OpenCompass, task settings were adjusted for compatibility with the LLaDA model under `lm-eval`.  
> Complete evaluation results will be released soon. -->

>  Results (evaluated) are from our framework, while results (reported) come from the original paper. All evaluation settings follow the configurations in the [LLaDA](https://github.com/ML-GSAI/LLaDA) repository, with minor adjustments. Placeholder entries (“–”) indicate results not yet evaluated; full results will be released soon.

<!-- <div align="center" style="min-width:1300px;"> -->

|               | MMLU | BBH | ARC&#8209;C | Hellaswag | TruthfulQA | WinoGrande | PIQA | GSM8K | Math | GPQA | HumanEval | MBPP | CEval | CMMLU |
|:----------------|:----:|:---:|:-----:|:-----------:|:-----------:|:------------:|:----:|:-----:|:----:|:----:|:-----------:|:----:|:------:|:------:|
| [`LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)(reported)| 0.659 | 0.497 | 0.459 | 0.705 | 0.461 | 0.748 | 0.736 | 0.703 | 0.314 | 0.252 | 0.354 | 0.400 | 0.705 | 0.699 |
| [`LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)(evaluated)| 0.658 | – | 0.457 | 0.693 | 0.456 | 0.707 | 0.706 | 0.704 | – | – | 0.323 | 0.388 | 0.702 | 0.699 |

<p align="center" style="color: #808080; font-size: 0.9em;">
Table 1. Evaluation results of 
<a href="https://huggingface.co/GSAI-ML/LLaDA-8B-Base" style="color: #808080; text-decoration: none;">
<code>LLaDA-8B-Base</code>
</a>.
</p>

|                 | MMLU | MMLU&#8209;Pro | ARC&#8209;C | Hellaswag | GSM8K | Math | GPQA | HumanEval | MBPP | 
|:----------------|:----:|:---------:|:-----:|:-----------:|:-----:|:----:|:----:|:-----------:|:----:|
| [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)(reported) | 0.655 | 0.370 | 0.885 | 0.746 | 0.694 | 0.319 | 0.333 | 0.494 | 0.410 |
| [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)(evaluated) | 0.673 | 0.362 | 0.866 | 0.767 | 0.811 | – | – | 0.650 | 0.702 |

<p align="center" style="color: #808080; font-size: 0.9em;">
Table 2. Evaluation results of 
<a href="https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct" style="color: #808080; text-decoration: none;">
<code>LLaDA-8B-Instruct</code>
</a>.
</p>
