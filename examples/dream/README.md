# Dream

> 📄 Paper: [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487) ｜ 💻 Code: [github.com/DreamLM/Dream](https://github.com/DreamLM/Dream)

Resources and examples for training (finetuning & pretraining) and evaluating diffusion language models **Dream**.

## Table of Contents
- [Setup](#setup)
- [Files](#files)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)

<!-- ## Setup
> [!IMPORTANT]  
> **Slurm users:** Update `scripts/train.slurm.sh` and `mkdir logs`: see [(optional) Slurm setup](/README.md/#optional-slurm-setup) for details.
> -->


##  Files
```
# pipeline modules relevant with Dream
dllm/pipelines/dream
├── __init__.py                     # Package initialization
├── models/
│   ├── configuration_dream.py      # Dream model configuration
│   ├── generation_utils.py         # Diffusion-based generation logic
│   ├── modeling_dream.py           # Core Dream model architecture
│   └── tokenization_dream.py       # Tokenizer implementation for Dream
├── eval.py                         # Evaluation module
├── sampler.py                      # Inference module
├── trainer.py                      # Training module (pretraining and SFT)
└── utils.py                        # Auxiliary utilities and helper functions

# example entry points for training / inference / evaluation
examples/dream
├── chat.py                         # Interactive inference example
├── eval.sh                         # Automatic evaluation example
├── sample.py                       # Inference example
├── pt.py                           # Pretraining example
├── README.md                       # Documentation (you are here)
└── sft.py                          # Supervised finetuning example
```
<!-- > [!NOTE]
>  We slightly modified [`modeling_dream.py`](/dllm/pipelines/dream/models/modeling_dream.py) so that the `model.forward()` supports 2-D attention masks. We recommend loading models with `dllm.utils.get_tokenizer`; otherwise `import dllm` before calling `AutoModel.from_pretrained` to ensure the correct models from `dllm` are used. 
> 
> We fixed bugs in `chat_template` and standardize `mask_token` through `dllm.utils.get_tokenizer`. If you use `AutoTokenizer`, keep in mind to set `chat_template` and `mask_token` appropriately yourselves. -->

## Training
  
> Read [Useful tips for training](/README.md/#useful-tips-for-training) and [(optional) Slurm setup](/README.md/#optional-slurm-setup) before training.

### SFT
For example, to SFT [`Dream-v0-Base-7B`](https://huggingface.co/Dream-org/Dream-v0-Base-7B) on the [`alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset for instruction following on 8 GPUs, run:
```shell
accelerate launch \
    --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/dream/sft.py \
    --model_name_or_path "Dream-org/Dream-v0-Base-7B" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 1024 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir "models/Dream-v0-Base-7B/alpaca"
```
If you are using slurm and want to train across, for example, 2 nodes (16 GPUs total), run:
```shell
sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/dream/sft.py" \
    --model_name_or_path "Dream-org/Dream-v0-Base-7B" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 1024 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir "models/Dream-v0-Base-7B/alpaca"
```

<!-- **Reproducing [Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Base-7B)**. We tried our best to reproduce Dream-v0-Instruct-7B by finetuning Dream-v0-Base-7B using our training pipeline on the public instruction-following dataset [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture): -->
#### Reproducing [`Dream-v0-Instruct-7B`](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) with SFT
We tried our best to reproduce [`Dream-v0-Instruct-7B`](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) by finetuning [`Dream-v0-Base-7B`](https://huggingface.co/Dream-org/Dream-v0-Base-7B) with SFT on the [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) dataset:

```shell
# Preprocessing SFT data (optional, but can avoid redundant preprocessing for multi-node training)
python dllm/tools/preprocess_sft_dataset.py \
    --model_name_or_path "Dream-org/Dream-v0-Base-7B" \
    --sft_map_fn_path "dllm.utils.default_sft_map_fn" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "data/sft/dream/tulu-3-sft-mixture" \
    --num_proc 64

# Train on 24*8=192 A100s with FSDP, take about 8 hours
sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/dream/sft.py" \
    --model_name_or_path "Dream-org/Dream-v0-Base-7B" \
    --dataset_args "data/sft/dream/tulu-3-sft-mixture" \
    --load_preprocessed_data True \
    --max_length 1024 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir "models/Dream-v0-Base-7B/tulu-3-sft-mixture"
```
<!-- [TODO] Training curves are on Wandb; checkpoints with evaluation results are available on Hugging Face. See the [Evaluation](#evaluation) section below for evaluation instructions. -->

### Pretraining

Pretrain on [`mlfoundations/dclm-baseline-1.0`](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) from scratch using 192 GPUs (24x8) and FSDP:
```shell
sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/dream/pt.py" \
    --model_name_or_path "Dream-org/Dream-v0-Base-7B" \
    --dataset_args "mlfoundations/dclm-baseline-1.0" \
    --max_length 1024 \
    --max_steps 2000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir "models/Dream-v0-Base-7B/dclm-baseline-1.0"
```

## Inference
We support batch inference for standard sampling and infilling:
```shell
python examples/dream/sample.py --model_name_or_path "Dream-org/Dream-v0-Instruct-7B"
```
We also support interactive multi-turn dialogue with visualization:
```shell
python examples/dream/chat.py --model_name_or_path "Dream-org/Dream-v0-Instruct-7B"
```

## Evaluation  
> Read [(optional) Evaluation setup](/README.md/#optional-evaluation-setup) before running evaluation. 

For example, to evaluate [`Dream-v0-Instruct-7B`](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) on [`gsm8k`](https://huggingface.co/datasets/openai/gsm8k) using 4 GPUs, run:
```shell
# Use model_args to adjust the sampler arguments for evaluation.
accelerate launch --num_processes 4 \
    dllm/pipelines/dream/eval.py \
    --tasks "gsm8k_cot" \
    --model "dream" \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,max_new_tokens=256,steps=256,temperature=0.1,top_p=0.9,alg=entropy,dtype=bfloat16,add_bos_token=False,escape_until=False"
```

To automatically evaluate [`Dream-v0-Base-7B`](https://huggingface.co/Dream-org/Dream-v0-Base-7B) and [`Dream-v0-Instruct-7B`](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) on all benchmarks, run:
```shell
bash examples/dream/eval.sh --model_name_or_path "Dream-org/Dream-v0-Instruct-7B" --instruct True
bash examples/dream/eval.sh --model_name_or_path "Dream-org/Dream-v0-Base-7B" --instruct False
```

### Evaluation results

>  Results (evaluated) are evaluated using our framework, while results (reported) come from the original [paper](https://arxiv.org/abs/2508.15487). All evaluation settings follow the configurations in the [Dream](https://github.com/DreamLM/Dream) repository, with minor adjustments.

|                 | MMLU | ARC&#8209;C | ARC&#8209;E | Hellaswag | WinoGrande | PIQA | GSM8K | Math | GPQA | HumanEval | MBPP | RACE |
|:----------------|:-------:|:-------:|:-----:|:-----------:|:------------:|:----:|:-----:|:----:|:----:|:-----------:|:----:|:------:|
| [`Dream-v0-Base-7B`](https://huggingface.co/Dream-org/Dream-v0-Base-7B) (reported) | 69.5 | 59.9 | 83.9 | 73.3 | 74.8 | 75.8 | 77.2 | 39.6 | 36.6 | 57.9 | 56.2 | 44.7 | 
| [`Dream-v0-Base-7B`](https://huggingface.co/Dream-org/Dream-v0-Base-7B) (evaluated) | 69.8 | 59.3 | 83.7 | 73.1 | 72.9 | 73.0 | 69.6 | 38.3 | 35.5 | 45.8 | 57.2 | 43.0 | 


<p align="center" style="color: #808080; font-size: 0.9em;">
Table 1. Evaluation results of 
<a href="https://huggingface.co/Dream-org/Dream-v0-Base-7B" style="color: #808080; text-decoration: none;">
<code>Dream-v0-Base-7B</code>
</a>.
</p>

|  | MMLU | MMLU-Pro | GSM8K | Math | GPQA | HumanEval | MBPP | IFEval |
|:----------------|:----:|:---------:|:-----:|:----:|:----:|:-----------:|:----:|:----:|
| [`Dream-v0-Instruct-7B`](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B)(reported)  | 67.0 | 43.3 | 81.0 | 39.2 | 33.0 | 55.5 | 58.8 | 62.5 |
| [`Dream-v0-Instruct-7B`](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B)(evaluated) | 68.2 | 43.0 | 82.0 | 39.9 | 32.4 | 59.1 | 58.2 | 62.3 |

<p align="center" style="color: #808080; font-size: 0.9em;">
Table 2. Evaluation results of 
<a href="https://huggingface.co/Dream-org/Dream-v0-Instruct-7B" style="color: #808080; text-decoration: none;">
<code>Dream-v0-Instruct-7B</code>
</a>.
</p>
