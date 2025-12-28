```shell
srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:000 python dllm/pipelines/andi/convert.py --model_name_or_path "Qwen/Qwen3-0.6B" --output_dir "models/andi/Qwen3-0.6B"

srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:000 python dllm/pipelines/andi/convert.py --model_name_or_path "Qwen/Qwen3-0.6B" --output_dir "models/andi/Qwen3-0.6B-random" --random_init
```


SFT
```shell
WANDB_MODE=online WANDB_PROJECT="dllm-andi" sbatch --nodes=1 --gres=gpu:8 --job-name=dllm-andi scripts/train.slurm.sh \
    --accelerate_config "zero2" \
    --script_path "examples/andi/mdlm/sft.py" \
    --model_name_or_path "models/andi/Qwen3-0.6B" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 512 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --loss_weight_type "scheduler" \
    --loss_normalization_type "token" \
    --output_dir "models/andi/Qwen3-0.6B/ar+mdlm/scheduler+token/alpaca"

WANDB_MODE=online WANDB_PROJECT="dllm-andi" sbatch --nodes=1 --gres=gpu:8 --job-name=dllm-andi scripts/train.slurm.sh \
    --accelerate_config "zero2" \
    --script_path "examples/andi/mdlm/sft.py" \
    --model_name_or_path "models/andi/Qwen3-0.6B" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 512 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --loss_weight_type "scheduler" \
    --loss_normalization_type "token" \
    --ar_loss_scale 0.0 \
    --output_dir "models/andi/Qwen3-0.6B/mdlm/scheduler+token/alpaca"

# WANDB_MODE=online WANDB_PROJECT="dllm-andi" sbatch --nodes=1 --gres=gpu:8 --job-name=dllm-andi scripts/train.slurm.sh \
#     --accelerate_config "zero2" \
#     --script_path "examples/andi/mdlm/sft.py" \
#     --model_name_or_path "models/andi/Qwen3-0.6B" \
#     --dataset_args "tatsu-lab/alpaca" \
#     --max_length 512 \
#     --num_train_epochs 10 \
#     --learning_rate 1e-4 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --loss_weight_type "scheduler" \
#     --loss_normalization_type "token" \
#     --diff_loss_scale 0.0 \
#     --output_dir "models/andi/Qwen3-0.6B/ar/scheduler+token/alpaca"
```

PT
```shell
WANDB_MODE=online WANDB_PROJECT="dllm-andi" sbatch --nodes=1 --gres=gpu:8 --job-name=dllm-andi scripts/train.slurm.sh \
    --accelerate_config "zero2" \
    --script_path "examples/andi/mdlm/pt.py" \
    --model_name_or_path "models/andi/Qwen3-0.6B" \
    --dataset_args "dylanebert/openwebtext[train:7_900_000,test:100_000]" \
    --max_length 512 \
    --max_steps 100_000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --loss_weight_type "scheduler" \
    --loss_normalization_type "token" \
    --output_dir "models/andi/Qwen3-0.6B/ar+mdlm/scheduler+token/openwebtext"

WANDB_MODE=online WANDB_PROJECT="dllm-andi" sbatch --nodes=1 --gres=gpu:8 --job-name=dllm-andi scripts/train.slurm.sh \
    --accelerate_config "zero2" \
    --script_path "examples/andi/mdlm/pt.py" \
    --model_name_or_path "models/andi/Qwen3-0.6B" \
    --dataset_args "dylanebert/openwebtext[train:7_900_000,test:100_000]" \
    --max_length 512 \
    --max_steps 100_000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --loss_weight_type "scheduler" \
    --loss_normalization_type "token" \
    --ar_loss_scale 0.0 \
    --output_dir "models/andi/Qwen3-0.6B/mdlm/scheduler+token/openwebtext"
```

PT (random)
```shell
WANDB_MODE=online WANDB_PROJECT="dllm-andi" sbatch --nodes=1 --gres=gpu:8 --job-name=dllm-andi scripts/train.slurm.sh \
    --accelerate_config "zero2" \
    --script_path "examples/andi/mdlm/pt.py" \
    --model_name_or_path "models/andi/Qwen3-0.6B-random" \
    --dataset_args "dylanebert/openwebtext[train:7_900_000,test:100_000]" \
    --max_length 512 \
    --max_steps 100_000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --loss_weight_type "scheduler" \
    --loss_normalization_type "token" \
    --output_dir "models/andi/Qwen3-0.6B-random/ar+mdlm/scheduler+token/openwebtext"

WANDB_MODE=online WANDB_PROJECT="dllm-andi" sbatch --nodes=1 --gres=gpu:8 --job-name=dllm-andi scripts/train.slurm.sh \
    --accelerate_config "zero2" \
    --script_path "examples/andi/mdlm/pt.py" \
    --model_name_or_path "models/andi/Qwen3-0.6B-random" \
    --dataset_args "dylanebert/openwebtext[train:7_900_000,test:100_000]" \
    --max_length 512 \
    --max_steps 100_000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --loss_weight_type "scheduler" \
    --loss_normalization_type "token" \
    --ar_loss_scale 0.0 \
    --output_dir "models/andi/Qwen3-0.6B-random/mdlm/scheduler+token/openwebtext"
```

Sample with different modes
```shell
srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:000 python -u examples/andi/mdlm/sample.py
```
