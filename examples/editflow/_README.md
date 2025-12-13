# Edit Flows

> **Reference**
> 📄 Paper: [Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/abs/2506.09018) 

This directory provides an educational reference for training EditFlow models. It demonstrates how to adapt open-weight DLLMs—such as [LLaDA](https://arxiv.org/abs/2502.09992) and [Dream](https://arxiv.org/abs/2508.15487)—to support *insertion*, *deletion*, beyond the standard *substitution*(`mask`->`tokens`) operations. It also includes examples for training (pretraining and finetuning) EditFlow models from scratch.

> [!NOTE]
> - Examples are available for both LLaDA and Dream, but this README focuses on adapting open-weight LLaDA for edit operations ([`adapt_llada.py`](/examples/editflow/adapt_llada.py)) and reusing its architecture for training from scratch ([`pt_llada.py`](/examples/editflow/pt_llada.py) -> [`sft_llada.py`](/examples/editflow/sft_llada.py)).
> - While `EditFlowCollator` supports custom `x0`, this README uses a fixed-length (128) masks as `x0`. The trained model samples text by replacing masks, deleting redundant ones, and inserting tokens as needed. To change the default `x0` distribution (e.g., empty sequences for [OneFlow](https://arxiv.org/abs/2510.03506)-like insertion-only sampling), pass `--x0_sampler "empty"`.

## Table of Contents
- [Setup](#setup)
- [Files overview](#files-overview)
- [Training](#training)
    - [Adapting LLaDA-8B-Instruct to support insertion and deletion](#adapting-llada-8b-instruct-to-support-insertion-and-deletion)
    - [Pretraining & Finetuning from scratch](#pretraining--finetuning-from-scratch)
- [Sampling](#sampling)
- [Acknowledgement](#acknowledgement)

## Setup
> [!IMPORTANT]  
> **Slurm users:** Update `scripts/train.slurm.sh` and `mkdir logs`: see [(optional) Slurm setup](/README.md/#optional-slurm-setup) for details.

##  Files overview
```
dllm/pipelines/editflow
├── __init__.py                 # Package initialization
├── models
│   ├── dream
│   │   └── modelling_dream.py  # EditFlowDream: architecture based on Dream
│   └── llada
│       └── modelling_llada.py  # EditFlowLLaDA: architecture based on LLaDA
├── trainer.py
└── utils.py

# example entry point for training / sampling
examples/editflow
├── adapt_dream.py              # Example of adapting Dream for EditFlow directly
├── adapt_llada.py              # Example of adapting LLaDA for EditFlow directly
├── sample.py                 # Sampling example
├── pt_dream.py                 # EditFlowDream pretraining example
├── pt_llada.py                 # EditFlowLLaDA pretraining example
├── pt.py                       # Pretraining function
├── README.md                   # Documentation (you are here)
├── sft_dream.py                # EditFlowDream SFT example
├── sft_llada.py                # EditFlowLLaDA SFT example
└── sft.py                      # Supervised finetuning function
```

## Training

### Adapting [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) to support *insertion* and *deletion*

The original LLaDA model samples text by iteratively substituting the given `<mask>` tokens to real tokens. 

<p align="center">
  <img src="https://github.com/ML-GSAI/LLaDA/blob/main/imgs/example_gradio.gif" alt="LLaDA demo" width="80%">
</p>
<p align="center"><em>Figure: Example Gradio demo for LLaDA.</em></p>

However, LLaDA supports only substitution. This example shows how to adapt it so that, during decoding, the model can not only replace fixed-length masks (e.g., 128 tokens) with real text but also insert new tokens and delete unnecessary masks adaptively:

```shell
accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/editflow/adapt_llada.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --lm_head_key "model.transformer.ff_out" \
    --init_editflow_from_src True \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/EditFlow-LLaDA-8B-Instruct-Adapt/tulu-3-sft-mixture" \
    --x0_sampler "masks[length:128]" \
    --max_length 1024 \
    --num_train_epochs 4 \
    --learning_rate 5e-5
```

If you are using slurm and want to train across, for example, four nodes (32 GPUs total), run:
```shell
sbatch --nodes=4 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/editflow/adapt_llada.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --lm_head_key "model.transformer.ff_out" \
    --init_editflow_from_src True \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/EditFlow-LLaDA-8B-Instruct-Adapt/tulu-3-sft-mixture" \
    --x0_sampler "masks[length:128]" \
    --max_length 1024 \
    --num_train_epochs 4 \
    --learning_rate 5e-5
```

After training, you can use the [sample.py](/examples/editflow/sample.py) scripts to provide a visualized decoding trace to see how the model performs *insertion* and *deletion* beyond regular mask *substitutions*. See [Sampling](#sampling) for details.


### Pretraining & Finetuning from scratch
You can also train an EditFlow model from scratch (pretrain → SFT) without adapting an existing DLLM.

Pretrain on a subset of [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) using 192 GPUs (24x8) and FSDP:

```shell
sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/editflow/pt_llada.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "mlfoundations/dclm-baseline-1.0" \
    --output_dir "models/EditFlow-LLaDA-8B-Base/dclm-baseline-1.0" \
    --x0_sampler "masks[length:128]" \
    --max_length 1024 \
    --max_steps 2000 \
    --learning_rate 3e-4
```

Finetune on a subset of [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) using 8 GPUS and FSDP for better instruction following:

```shell
# you can also run locally with `accelerate ...`
sbatch --nodes=1 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/editflow/sft_llada.py" \
    --model_name_or_path "models/EditFlow-LLaDA-8B-Base/dclm-baseline-1.0/checkpoint-final" \
    --dataset_args "allenai/tulu-3-sft-mixture[train:10000,test:1000]" \
    --output_dir "models/EditFlow-LLaDA-8B-Base/dclm-baseline-1.0" \
    --x0_sampler "masks[length:128]" \
    --max_length 1024 \
    --num_train_epochs 4 \
    --learning_rate 5e-5
```

## Sampling

After training, you can visualize how the model performs mask substitution, insertion, and deletion during sampling with [sample.py](/examples/editflow/sample.py). Inserted tokens appear <span style="color:blue; font-weight:bold">blue</span>, and tokens substituted from `<mask>` appear <span style="color:black; font-weight:bold">black</span>, and deleted tokens are shown with a strikethrough before they disappear.

```shell
# Sample a long sequence to visualize insertions after 128 <mask> tokens
python examples/editflow/sample.py \
  --model_name_or_path "models/EditFlow-LLaDA-8B-Instruct-Adapt/tulu-3-sft-mixture/checkpoint-final" \
  --tau 0.02 --mask_length 128 --seed 7070 \
  --prompt "write a romantic story" --make_gif

# Sample a short sequence to visualize deletions after 128 <mask> tokens
python examples/editflow/sample.py \
  --model_name_or_path "models/EditFlow-LLaDA-8B-Instruct-Adapt/tulu-3-sft-mixture/checkpoint-final" \
  --tau 0.02 --mask_length 128 --seed 7070 \
  --prompt "write a single-sentence romantic story" --make_gif
```

<p align="center">
  <img src="/examples/editflow/assets/deletion.gif" alt="EditFlow deletion demo" width="95%">
</p>
<p align="center"><em>Figure: Deletion & Substitution trace</code></em></p>

<p align="center">
  <img src="/examples/editflow/assets/insertion.gif" alt="LLaDA demo" width="95%">
</p>
<p align="center"><em>Figure: Inserction & Substitution trace</em></p>

## Acknowledgement

This Edit Flows implementation is inspired by https://github.com/TheMatrixMaster/edit-flows-demo.
