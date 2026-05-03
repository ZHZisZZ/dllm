#!/usr/bin/env bash
# Info-Gain LLaDA evaluation (mirrors examples/fastdllm/llada/eval.sh structure).

export PYTHONPATH=.:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL

model_name_or_path="GSAI-ML/LLaDA-8B-Instruct"
instruct=True
num_gpu=1
max_new_tokens=256
threshold="0.8"
candidate_number=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path) model_name_or_path="$2"; shift 2 ;;
    --instruct) instruct="$2"; shift 2 ;;
    --num_gpu) num_gpu="$2"; shift 2 ;;
    --max_new_tokens) max_new_tokens="$2"; shift 2 ;;
    --threshold) threshold="$2"; shift 2 ;;
    --candidate_number) candidate_number="$2"; shift 2 ;;
    *) echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

if [ "$instruct" = "True" ]; then
    echo ">>> Running in INSTRUCT mode"
    llada_args="--model llada --apply_chat_template"
    infogain_args="--model infogain_llada --apply_chat_template"
else
    echo ">>> Running in BASE mode"
    llada_args="--model llada"
    infogain_args="--model infogain_llada"
fi

IG_ARGS="pretrained=${model_name_or_path},use_cache=none,threshold=${threshold},candidate_number=${candidate_number},position_temperature=0.2,variant=info_gain,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[]"

# GSM8K — baseline vs Info-Gain (no KV cache)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${llada_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/infogain/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${infogain_args} \
    --model_args "${IG_ARGS}"

# Info-Gain + Fast-dLLM prefix cache (KV path; not Info-Gain inner objective)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/infogain/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${infogain_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[]"
