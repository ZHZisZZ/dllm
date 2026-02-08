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
model_name_or_path="dllm-collection/ModernBERT-large-chat-v0.1"
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
common_args="--model bert --apply_chat_template"  # BERT model uses chat template by default

# =======================
# BERT Chat Tasks
# =======================

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/bert/eval.py \
    --tasks gsm8k_bert --num_fewshot 8 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=32"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/bert/eval.py \
    --tasks bbh --num_fewshot 3 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=32"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/bert/eval.py \
    --tasks minerva_math --num_fewshot 4 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=32"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/bert/eval.py \
    --tasks ceval-valid --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},mc_num=128"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/bert/eval.py \
    --tasks mmlu --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},mc_num=128"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/bert/eval.py \
    --tasks winogrande --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},mc_num=128"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/bert/eval.py \
    --tasks hellaswag --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},mc_num=128"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/bert/eval.py \
    --tasks cmmlu --num_fewshot 5 ${common_args} \
    --model_args "pretrained=${model_name_or_path},mc_num=128"
