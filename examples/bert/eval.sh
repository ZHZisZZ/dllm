 #!/usr/bin/env bash
# ===== Mandatory for proper import and evaluation =====
export PYTHONPATH=.:$PYTHONPATH             
export HF_ALLOW_CODE_EVAL=1                 # Allow code evaluation
export HF_DATASETS_TRUST_REMOTE_CODE=True   # For cmmlu dataset

# ===== Optional but recommended for stability and debugging =====
export PYTHONBREAKPOINT=0                   # Disable interactive breakpoints
export NCCL_ASYNC_ERROR_HANDLING=1          # Enable async error handling for multi-GPU communication to avoid deadlocks
export NCCL_DEBUG=warn                      # Show NCCL warnings for better diagnosis without flooding logs
export TORCH_DISTRIBUTED_DEBUG=DETAIL       # Provide detailed logging for PyTorch distributed debugging

# ===== Basic Settings =====
model_name_or_path="dllm-collection/ModernBERT-large-chat-v0"
num_gpu=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)
      model_name_or_path="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
  esac
done

# ===== Common arguments =====
common_args="--model bert --apply_chat_template"  # BERT model is default to use chat template

# =======================
# BERT Instruct (Chat) Tasks
# =======================

accelerate launch --num_processes ${num_gpu} dllm/pipelines/bert/eval.py \
    --tasks hellaswag_gen --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=128,steps=128,block_length=128"

accelerate launch --num_processes ${num_gpu} dllm/pipelines/bert/eval.py \
    --tasks mmlu_generative --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=128,steps=128,block_length=128"

accelerate launch --num_processes ${num_gpu} dllm/pipelines/bert/eval.py \
    --tasks mmlu_pro --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=256,steps=256,block_length=256"

accelerate launch --num_processes ${num_gpu} dllm/pipelines/bert/eval.py \
    --tasks arc_challenge_chat --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=128,steps=128,block_length=128"

accelerate launch --num_processes ${num_gpu} dllm/pipelines/bert/eval.py \
    --tasks winogrande --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=128,steps=128,block_length=128"
