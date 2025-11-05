# Edit Flows - BERT

> 📄 Paper: [Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/abs/2506.09018) 


## Warmup

In this section, we show toy examples of pretraining and SFTing [`ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) on small datasets to generate text with EditFlow.
You can use any BERT model instead for example, by `--model_name_or_path "FacebookAI/roberta-large"`.

### Pretrain

To train [`ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) on the [`tiny-shakespeare`](https://huggingface.co/datasets/Trelis/tiny-shakespeare) dataset, run:
```shell
PYTHONPATH=. accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/editflow/bert/pt.py \
    --model_name_or_path "answerdotai/ModernBERT-large" \
    --dataset_args "Trelis/tiny-shakespeare" \
    --text_field "Text" \
    --insert_eos False \
    --max_length 128 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --save_steps 0.1 \
    --x0_sampler "masks[length:64]" \
    --output_dir "models/EditFlow/ModernBERT-large/tiny-shakespeare"
```

To run inference with the model:
```shell
PYTHONPATH=. python examples/editflow/generate.py \
    --model_name_or_path "models/EditFlow/ModernBERT-large/tiny-shakespeare/checkpoint-final" \
    --tau 0.01 --mask_length 64 --seed 42 --make_gif

# see `decode_trace.gif`
```


### SFT
To train [`ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) on the [`alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset, run:
```shell
PYTHONPATH=. accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/editflow/bert/sft.py \
    --model_name_or_path "answerdotai/ModernBERT-large" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 512 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --save_steps 0.1 \
    --x0_sampler "masks[length:64]" \
    --output_dir "models/EditFlow/ModernBERT-large/alpaca"
```

To run inference with the model:
```shell
PYTHONPATH=. python examples/editflow/generate.py \
    --model_name_or_path "models/EditFlow/ModernBERT-large/alpaca/checkpoint-final" \
    --prompt "Could you please write a poem for me?" --tau 0.01 --mask_length 64 --seed 42 --make_gif

# see `decode_trace.gif`
```

<!-- ```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/editflow/bert/sft.py \
    --model_name_or_path "answerdotai/ModernBERT-large" \
    --dataset_args "allenai/tulu-3-sft-mixture|HuggingFaceTB/smoltalk" \
    --max_length 1024 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 48 \
    --save_steps 0.1 \
    --x0_sampler "masks[length:64]" \
    --output_dir "models/EditFlow/ModernBERT-large/tulu-3-smoltalk/epochs-10-bs-384-len-1024"
``` -->
