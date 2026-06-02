#!/usr/bin/env bash
# Info-Gain Dream evaluation (mirrors examples/fastdllm/dream/eval.sh, adds alg=info_gain).

export PYTHONPATH=.:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL

model_name_or_path="Dream-org/Dream-v0-Base-7B"
instruct=False
num_gpu=1
max_new_tokens=256
block_size=32
threshold="0.8"
candidate_number=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path) model_name_or_path="$2"; shift 2 ;;
    --instruct) instruct="$2"; shift 2 ;;
    --num_gpu) num_gpu="$2"; shift 2 ;;
    --max_new_tokens) max_new_tokens="$2"; shift 2 ;;
    --block_size) block_size="$2"; shift 2 ;;
    --threshold) threshold="$2"; shift 2 ;;
    --candidate_number) candidate_number="$2"; shift 2 ;;
    *) echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

if [ "$instruct" = "True" ]; then
    echo ">>> Running in INSTRUCT mode"
    dream_args="--model dream --apply_chat_template"
    infogain_args="--model infogain_dream --apply_chat_template"
else
    echo ">>> Running in BASE mode"
    dream_args="--model dream"
    infogain_args="--model infogain_dream"
fi

BASE_DREAM="pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},alg=entropy,dtype=bfloat16,add_bos_token=True"

IG_DREAM="pretrained=${model_name_or_path},use_cache=none,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=${block_size},alg=info_gain,threshold=${threshold},candidate_number=${candidate_number},position_temperature=0.2,info_gain_variant=info_gain,temperature=0.0,top_p=0.9,dtype=bfloat16,add_bos_token=True"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/dream/eval.py \
    --tasks gsm8k --num_fewshot 5 ${dream_args} \
    --model_args "${BASE_DREAM}"

accelerate launch --num_processes "${num_gpu}" dllm/pipelines/infogain/dream/eval.py \
    --tasks gsm8k --num_fewshot 5 ${infogain_args} \
    --model_args "${IG_DREAM}"
