#!/usr/bin/env bash
# ===== Mandatory for proper import and evaluation =====
export PYTHONPATH=.:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1                 # Allow code evaluation
export HF_DATASETS_TRUST_REMOTE_CODE=True   # For cmmlu dataset

# ===== Optional but recommended for stability and debugging =====
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Enable async error handling for multi-GPU communication to avoid deadlocks
export NCCL_DEBUG=warn                      # Show NCCL warnings for better diagnosis without flooding logs
export TORCH_DISTRIBUTED_DEBUG=DETAIL       # Provide detailed logging for PyTorch distributed debugging

# ===== Input Arguments =====
model_name_or_path="GSAI-ML/LLaDA-8B-Instruct"
instruct=True
num_gpu=1
max_new_tokens=256
use_cache="prefix"
threshold="0.9"
factor="1.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)
      model_name_or_path="$2"; shift 2 ;;
    --instruct)
      instruct="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
    --use_cache)
      use_cache="$2"; shift 2 ;;
    --threshold)
      threshold="$2"; shift 2 ;;
    *)
      echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

# ===== Conditional Configurations =====
if [ "$instruct" = "True" ]; then
    echo ">>> Running in INSTRUCT mode"
    common_args="--model fastdllm_llada --apply_chat_template"
else
    echo ">>> Running in BASE mode"
    common_args="--model fastdllm_llada"
fi

# =======================
# GSM8K Task Evaluation
# =======================

# Baseline (29.35s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"

# Parallel threshold (12.60s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=none,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"

# Prefix cache (9.25s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"

# Dual cache (9.53s/it))
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"
    
# Prefix cache + Parallel threshold (4.39s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"

# Dual cache + Parallel threshold(4.83s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"

# Prefix cache + Parallel factor(3.68s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,factor=${factor},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"

# Dual cache + Parallel factor(3.78s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,factor=${factor},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"

# ===========================
# Humaneval Task Evaluation
# ===========================

# Baseline (12.03s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Parallel threshold (4.20s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=none,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Prefix cache + Parallel threshold (2.95/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Prefix cache + Parallel factor (2.39)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,factor=${factor},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Prefix cache (7.33s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Dual cache (9.19s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code
  
# Dual cache + Parallel threshold (3.90s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Dual cache + Parallel factor (3.15s/it)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,factor=${factor},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code
