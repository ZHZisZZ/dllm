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


# ===== Input Arguments =====
model_name_or_path="GSAI-ML/LLaDA-8B-Instruct"
instruct=True
num_gpu=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)
      model_name_or_path="$2"; shift 2 ;;
    --instruct)
      instruct="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
  esac
done

# ===== Conditional Configurations =====
if [ "$instruct" = "True" ]; then
    echo ">>> Running in INSTRUCT mode"
    common_args="--model llada --apply_chat_template"
else
    echo ">>> Running in BASE mode"
    common_args="--model llada"
fi


# =======================
# Generation Tasks
# =======================

if [ "$instruct" = "True" ]; then
    # Instruct Generation Tasks
    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks gsm8k_cot --num_fewshot 8 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks bbh --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks minerva_math --num_fewshot 4 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks humaneval_instruct --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks mbpp_llada_instruct --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

else
    # Base Generation Tasks
    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks gsm8k --num_fewshot 8 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks bbh --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks minerva_math --num_fewshot 4 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks humaneval --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks mbpp --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"
fi


# =======================
# Likelihood Tasks
# =======================

if [ "$instruct" = "True" ]; then
    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks mmlu_generative --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=3,steps=3,block_length=3,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks mmlu_pro --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=256,steps=256,block_length=256,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks hellaswag_gen --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=3,steps=3,block_length=3,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks arc_challenge_chat --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=5,steps=5,block_length=5,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks gpqa_n_shot_gen --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=32,steps=32,block_length=32,cfg=0.0"

else
    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks gpqa_main_n_shot --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.5"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks truthfulqa_mc2 --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=2.0"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks arc_challenge --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.5"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks hellaswag --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.5"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks winogrande --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks piqa --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.5"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks mmlu --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks cmmlu --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.0"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/llada/eval.py \
        --tasks ceval-valid --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.0"
fi
