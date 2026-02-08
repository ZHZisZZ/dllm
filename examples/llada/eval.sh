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
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)
      model_name_or_path="$2"; shift 2 ;;
    --instruct)
      instruct="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
    *) 
      echo "Error: Unknown argument: $1"; exit 1 ;;
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
# LLaDA-1.0 Tasks
# =======================

if [ "$instruct" = "True" ]; then
    # Instruct Tasks
    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks mmlu_generative --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=3,steps=3,block_size=3,cfg=0.0,suppress_tokens=[],begin_suppress_tokens=[]"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks mmlu_pro --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=256,cfg=0.0,suppress_tokens=[],begin_suppress_tokens=[]"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks hellaswag_gen --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=3,steps=3,block_size=3,cfg=0.0,suppress_tokens=[],begin_suppress_tokens=[]"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks arc_challenge_chat --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=512,steps=512,block_size=512,cfg=0.0,suppress_tokens=[],begin_suppress_tokens=[]"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks gpqa_diamond_generative_n_shot --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=64,steps=64,block_size=64,cfg=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks gsm8k_cot --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=512,steps=512,block_size=512,cfg=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks minerva_math --num_fewshot 4 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=512,steps=512,block_size=512,cfg=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks humaneval_instruct_llada --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=512,steps=512,block_size=512,cfg=0.0,suppress_tokens=[126081],begin_suppress_tokens=[]" \
        --confirm_run_unsafe_code

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks mbpp_instruct_llada --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,block_size=256,cfg=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
        --confirm_run_unsafe_code

else
    # Base Tasks
    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks gsm8k --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=1024,steps=1024,block_size=1024,cfg=0.0"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks bbh --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=1024,steps=1024,block_size=1024,cfg=0.0"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks minerva_math --num_fewshot 4 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=1024,steps=1024,block_size=1024,cfg=0.0"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks humaneval --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=1024,steps=1024,block_size=1024,cfg=0.0" \
        --confirm_run_unsafe_code

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks mbpp --num_fewshot 3 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=1024,steps=1024,block_size=1024,cfg=0.0" \
        --confirm_run_unsafe_code

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks gpqa_main_n_shot --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=128,cfg=0.5"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks truthfulqa_mc2 --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=128,cfg=2.0"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks arc_challenge --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=128,cfg=0.5"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks hellaswag --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=128,cfg=0.5"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks winogrande --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=128,cfg=0.0"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks piqa --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=128,cfg=0.5"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks mmlu --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,cfg=0.0"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks cmmlu --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,cfg=0.0"

    accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
        --tasks ceval-valid --num_fewshot 5 ${common_args} \
        --model_args "pretrained=${model_name_or_path},mc_num=1,cfg=0.0"
fi
