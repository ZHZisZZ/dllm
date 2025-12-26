<!-- Work in progress. 

Please see [`examples/editflow/bert/README.md`](/examples/editflow/bert/README.md) for examples of finetuning BERT with EditFlow. -->


# Edit Flows

> 📄 Paper: [Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/abs/2506.09018) 


## Setup

> (Optional) If your source model is **not already supported** in [`dllm/pipelines/editflow/models`](/dllm/pipelines/editflow/models) for Edit Flows:
> Modify the original modeling file to support rate and operation heads. See [`modelling_modernbert.py`](/dllm/pipelines/editflow/models/bert/modelling_modernbert.py#L77-L108) for an example.

Before training, save the source bidirectional-attention model with the architectural changes required for Edit Flows (rate and operation heads).
For example, to convert [`answerdotai/ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) while preserving its original weights:
```shell
python dllm/pipelines/editflow/convert.py \
  --model_name_or_path "answerdotai/ModernBERT-large" \
  --output_dir "models/editflow/ModernBERT-large"

# LLaDA
# python dllm/pipelines/editflow/convert.py \
#   --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
#   --output_dir "models/editflow/LLaDA-8B-Base"
```

<details>
<summary>Example of preparing an autoregressive model (e.g., Qwen) for Edit Flows</summary>

Autoregressive models must first be converted to use bidirectional attention, then augmented with rate and operation heads.

```shell
# Convert to bidirectional attention
python dllm/pipelines/a2d/convert.py \
  --model_name_or_path "Qwen/Qwen3-0.6B" \
  --output_dir "models/a2d/Qwen3-0.6B"

# Add rate and operation heads
python dllm/pipelines/editflow/convert.py \
  --model_name_or_path "models/a2d/Qwen3-0.6B" \
  --output_dir "models/editflow/Qwen3-0.6B"
```

</details>

## Training

> Read [Useful tips for training](/README.md/#useful-tips-for-training) and [(optional) Slurm setup](/README.md/#optional-slurm-setup) before training.
>
> When training larger source models (e.g., [`LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) or [`Dream-v0-Base-7B`](https://huggingface.co/Dream-org/Dream-v0-Base-7B)), consider 
> using `--config_file scripts/accelerate_configs/fsdp.yaml`, reducing `per_device_train_batch_size` or enabling `--load_in_4bit True --lora True` to reduce VRAM usage.

### Continual Pretraining

To train [`ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) on the [`tiny-shakespeare`](https://huggingface.co/datasets/Trelis/tiny-shakespeare) dataset, run (on 1 GPU):

```shell
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/editflow/pt.py \
    --model_name_or_path "models/editflow/ModernBERT-large" \
    --dataset_args "Trelis/tiny-shakespeare" \
    --text_field "Text" \
    --max_length 128 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --x0_sampler "masks[length:64]" \
    --output_dir "models/editflow/ModernBERT-large/tiny-shakespeare"
```

To sample from the model interactively:
```shell
# Enter a prompt (e.g., "First citizen: Before we proceed any further, hear me speak."),
# or press Enter to let the model generate text from scratch.
python -u examples/editflow/chat.py \
    --model_name_or_path "models/editflow/ModernBERT-large/tiny-shakespeare/checkpoint-final" \
    --chat_template False
```

<details>
<summary>Example of pretraining on a larger dataset (OpenWebText) in streaming mode</summary>

```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/editflow/pt.py \
    --model_name_or_path "models/editflow/ModernBERT-large" \
    --dataset_args "dylanebert/openwebtext" \
    --text_field "text" \
    --streaming True \
    --max_length 512 \
    --max_steps 20000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --eval_strategy "no" \
    --x0_sampler "masks[length:64]" \
    --output_dir "models/editflow/ModernBERT-large/openwebtext"
```

To sample from the model interactively:
```shell
# Enter a prompt (e.g., "Lebron James is"),
# or press Enter to let the model generate text from scratch.
python -u examples/editflow/chat.py \
    --model_name_or_path "models/editflow/ModernBERT-large/openwebtext/checkpoint-final" \
    --chat_template False
```

</details>


### SFT

To train [`ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) on the [`alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset, run (on 8 GPUs):
```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/editflow/sft.py \
    --model_name_or_path "models/editflow/ModernBERT-large" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 512 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --x0_sampler "masks[length:64]" \
    --output_dir "models/editflow/ModernBERT-large/alpaca"
```

To chat with the model:
```shell
python -u examples/editflow/chat.py \
    --model_name_or_path "models/editflow/ModernBERT-large/alpaca/checkpoint-final"
```