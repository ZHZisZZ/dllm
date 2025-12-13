#!/usr/bin/env bash
# ===== Mandatory for proper import and evaluation =====
export PYTHONPATH=.:$PYTHONPATH             
export HF_ALLOW_CODE_EVAL=1                 # Allow code evaluation
export HF_DATASETS_TRUST_REMOTE_CODE=True   # For cmmlu dataset

# ===== Optional but recommended for stability and debugging =====
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Enable async error handling for multi-GPU communication to avoid deadlocks
export NCCL_DEBUG=warn                      # Show NCCL warnings for better diagnosis without flooding logs
export TORCH_DISTRIBUTED_DEBUG=DETAIL       # Provide detailed logging for PyTorch distributed debugging

# ===== Basic Settings =====
model_name_or_path="dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1"
num_gpu=1
model_type="normal"   # normal | coder

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)
      model_name_or_path="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
    --model_type)
      model_type="$2"; shift 2 ;;
    *)
      echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

echo ">>> model_type: ${model_type}"
echo ">>> model_name_or_path: ${model_name_or_path}"

# ===== Common arguments =====
common_args="--model bd3lm --apply_chat_template"

# =========================================================
# If coder model → Only run HumanEval + MBPP
# =========================================================

if [[ "$model_type" == "coder" ]]; then
    echo ">>> Running coder-model benchmark suite (HumanEval + MBPP only)"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/a2d/eval.py \
        --tasks humaneval_instruct --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=32,cfg=0.0" \
        --confirm_run_unsafe_code

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/a2d/eval.py \
        --tasks mbpp_instruct --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=32,cfg=0.0" \
        --confirm_run_unsafe_code

    exit 0
fi

# =========================================================
# Normal model → Run all tasks (full list)
# =========================================================

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/a2d/eval.py \
    --tasks mmlu_generative_dream --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=3,steps=3,block_size=32,cfg=0.0"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/a2d/eval.py \
    --tasks mmlu_pro --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=32,cfg=0.0"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/a2d/eval.py \
    --tasks hellaswag_gen --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=3,steps=3,block_size=32,cfg=0.0"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/a2d/eval.py \
    --tasks gsm8k_cot --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=32,cfg=0.0"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/a2d/eval.py \
    --tasks bbh --num_fewshot 3 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=32,cfg=0.0"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/a2d/eval.py \
    --tasks minerva_math --num_fewshot 4 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=32,cfg=0.0"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/a2d/eval.py \
    --tasks humaneval_instruct --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=32,cfg=0.0" \
    --confirm_run_unsafe_code

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/a2d/eval.py \
    --tasks mbpp_instruct --num_fewshot 3 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=32,cfg=0.0" \
    --confirm_run_unsafe_code
