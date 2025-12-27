```shell
python dllm/pipelines/andi/convert.py --model_name_or_path "Qwen/Qwen3-0.6B" --output_dir "models/andi/Qwen3-0.6B"
```


Train
```shell
# WANDB_MODE=online WANDB_PROJECT="dllm-andi" sbatch --nodes=1 --gres=gpu:8 --job-name=dllm-andi scripts/train.slurm.sh \
#     --accelerate_config "zero2" \
#     --script_path "examples/andi/mdlm/sft.py" \
#     --model_name_or_path "models/andi/Qwen3-0.6B" \
#     --dataset_args "tatsu-lab/alpaca" \
#     --max_length 512 \
#     --num_train_epochs 20 \
#     --learning_rate 1e-4 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --loss_weight_type "uniform" \
#     --output_dir "models/andi/Qwen3-0.6B/ar+mdlm/uniform/alpaca"

# WANDB_MODE=online WANDB_PROJECT="dllm-andi" sbatch --nodes=1 --gres=gpu:8 --job-name=dllm-andi scripts/train.slurm.sh \
#     --accelerate_config "zero2" \
#     --script_path "examples/andi/mdlm/sft.py" \
#     --model_name_or_path "models/andi/Qwen3-0.6B" \
#     --dataset_args "tatsu-lab/alpaca" \
#     --max_length 512 \
#     --num_train_epochs 20 \
#     --learning_rate 1e-4 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --loss_weight_type "uniform" \
#     --ar_loss_scale 0.0 \
#     --output_dir "models/andi/Qwen3-0.6B/mdlm/uniform/alpaca"

# WANDB_MODE=online WANDB_PROJECT="dllm-andi" sbatch --nodes=1 --gres=gpu:8 --job-name=dllm-andi scripts/train.slurm.sh \
#     --accelerate_config "zero2" \
#     --script_path "examples/andi/mdlm/sft.py" \
#     --model_name_or_path "models/andi/Qwen3-0.6B" \
#     --dataset_args "tatsu-lab/alpaca" \
#     --max_length 512 \
#     --num_train_epochs 20 \
#     --learning_rate 1e-4 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --loss_weight_type "uniform" \
#     --diff_loss_scale 0.0 \
#     --output_dir "models/andi/Qwen3-0.6B/ar/uniform/alpaca"

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
    --output_dir "models/andi/Qwen3-0.6B/ar+mdlm/scheduler/alpaca"

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
    --ar_loss_scale 0.0 \
    --output_dir "models/andi/Qwen3-0.6B/mdlm/scheduler/alpaca"

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
    --diff_loss_scale 0.0 \
    --output_dir "models/andi/Qwen3-0.6B/ar/scheduler/alpaca"
```

Sample with different modes
```shell
srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=24 --time=03:00:000 python -u examples/andi/mdlm/sample.py
```
