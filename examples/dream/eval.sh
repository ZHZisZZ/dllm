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
model_name_or_path="Dream-org/Dream-v0-Instruct-7B"
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
    common_args="--model dream --apply_chat_template"
else
    echo ">>> Running in BASE mode"
    common_args="--model dream"
fi


# =======================
# Generation / Instruct Tasks
# =======================

if [ "$instruct" = "True" ]; then
    # Instruct Tasks
    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks mmlu_generative --num_fewshot 4 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=128,max_length=128,steps=128,temperature=0.1,top_p=0.9,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks mmlu_pro --num_fewshot 4 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=128,max_length=128,steps=128,temperature=0.1,top_p=0.9,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks gsm8k_cot --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=256,max_length=256,steps=256,temperature=0.1,top_p=0.9,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks minerva_math --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.1,top_p=0.9,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks gpqa_main_n_shot --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=128,max_length=128,steps=128,temperature=0.0,top_p=1.0,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks humaneval_instruct_dream --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=768,max_length=768,steps=768,temperature=0.1,top_p=0.9,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks mbpp_instruct --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=1024,max_length=1024,steps=1024,temperature=0.1,top_p=0.9,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks ifeval --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=1280,max_length=1280,steps=1280,temperature=0.1,top_p=0.9,add_bos_token=true,escape_until=true"

else
    # Base Generation Tasks
    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks humaneval --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.2,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks gsm8k_cot --num_fewshot 8 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=256,max_length=256,steps=256,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks mbpp --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.2,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks minerva_math --num_fewshot 4 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks bbh --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"
fi


# =======================
# Likelihood Tasks (Base Only)
# =======================

if [ "$instruct" != "True" ]; then
    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks mmlu --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks arc_easy --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks arc_challenge --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks hellaswag --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks piqa --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks gpqa_main_n_shot --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks winogrande --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

    accelerate launch --num_processes ${num_gpu} dllm/pipelines/dream/eval.py \
        --tasks race --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"
fi
