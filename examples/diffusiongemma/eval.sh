#!/usr/bin/env bash
# ===== Mandatory for proper import and evaluation =====
export PYTHONPATH=.:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1                 # Allow code evaluation
export HF_DATASETS_TRUST_REMOTE_CODE=True   # For datasets that use remote code

# ===== Optional but recommended for stability and debugging =====
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Enable async error handling for multi-GPU communication to avoid deadlocks
export NCCL_DEBUG=warn                      # Show NCCL warnings for better diagnosis without flooding logs
export TORCH_DISTRIBUTED_DEBUG=DETAIL       # Provide detailed logging for PyTorch distributed debugging

# ===== Basic Settings =====
model_name_or_path="/mnt/weka/shrd/research/model/diffusiongemma-26B-A4B-it"
num_gpu=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)
      model_name_or_path="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
    *)
      echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

# ===== Common arguments =====
common_args="--model diffusiongemma --apply_chat_template"

# =======================
# DiffusionGemma Tasks
# =======================
# Use dllm/pipelines/diffusiongemma/eval.py and --model diffusiongemma
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/diffusiongemma/eval.py \
    --tasks gsm8k_cot --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=512,block_size=256,max_denoising_steps=48,entropy_bound=0.1,dtype=bfloat16"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/diffusiongemma/eval.py \
    --tasks minerva_math --num_fewshot 4 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=512,block_size=256,max_denoising_steps=48,entropy_bound=0.1,dtype=bfloat16"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/diffusiongemma/eval.py \
    --tasks humaneval_instruct --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=512,block_size=256,max_denoising_steps=48,entropy_bound=0.1,dtype=bfloat16" \
    --confirm_run_unsafe_code
