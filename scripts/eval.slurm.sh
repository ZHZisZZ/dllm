#!/usr/bin/env bash
#SBATCH --job-name=model-eval
#SBATCH --partition=mllm_safety
#SBATCH --quotatype=spot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=.logs/%x-%j.out
#SBATCH --error=.logs/%x-%j.err
#SBATCH --requeue

# ============================================================
# Unified Evaluation Configuration + Execution Script
# ============================================================


# ------------------------------------------------------------
# Declare associative arrays
# ------------------------------------------------------------
declare -A eval_llada_base_configs
declare -A eval_llada_instruct_configs

declare -A eval_dream_base_configs
declare -A eval_dream_instruct_configs

declare -A eval_bert_configs


# ============================================================
# ====================  LLaDA CONFIGS  ========================
# ============================================================
#   eval_llada_configs["<dataset>"]="num_fewshot|max_new_tokens|steps|block_size|seed|mc_num|cfg"
# ============================================================

# ---------- Base Generation ----------
eval_llada_base_configs["gsm8k"]="8|1024|1024|32|1234|128|0.0"
eval_llada_base_configs["bbh"]="3|1024|1024|32|1234|128|0.0"
eval_llada_base_configs["minerva_math"]="4|1024|1024|32|1234|128|0.0"
eval_llada_base_configs["humaneval"]="0|1024|1024|32|1234|128|0.0"
eval_llada_base_configs["mbpp"]="3|1024|1024|32|1234|128|0.0"

# ---------- Base Likelihood ----------
eval_llada_base_configs["gpqa_main_n_shot"]="5|1024|1024|1024|1234|128|0.5"
eval_llada_base_configs["truthfulqa_mc2"]="0|1024|1024|1024|1234|128|2.0"
eval_llada_base_configs["arc_challenge"]="0|1024|1024|1024|1234|128|0.5"
eval_llada_base_configs["hellaswag"]="0|1024|1024|1024|1234|128|0.5"
eval_llada_base_configs["winogrande"]="5|1024|1024|1024|1234|128|0.0"
eval_llada_base_configs["piqa"]="0|1024|1024|1024|1234|128|0.5"
eval_llada_base_configs["mmlu"]="5|1024|1024|1024|1234|1|0.0"
eval_llada_base_configs["cmmlu"]="5|1024|1024|1024|1234|1|0.0"
eval_llada_base_configs["ceval-valid"]="5|1024|1024|1024|1234|1|0.0"

# ---------- Instruct Generation ----------
eval_llada_instruct_configs["gsm8k_cot"]="8|1024|1024|32|1234|1|0.0"
eval_llada_instruct_configs["bbh"]="3|1024|1024|32|1234|1|0.0"
eval_llada_instruct_configs["minerva_math"]="4|1024|1024|32|1234|1|0.0"
eval_llada_instruct_configs["humaneval_instruct"]="0|1024|1024|32|1234|1|0.0"
eval_llada_instruct_configs["mbpp_llada_instruct"]="3|1024|1024|32|1234|1|0.0"

eval_llada_instruct_configs["mmlu_generative"]="0|3|3|3|1234|1|0.0"
eval_llada_instruct_configs["mmlu_pro"]="0|256|256|256|1234|1|0.0"
eval_llada_instruct_configs["hellaswag_gen"]="0|3|3|3|1234|1|0.0"
eval_llada_instruct_configs["arc_challarc_challenge_chatenge"]="0|5|5|5|1234|1|0.0"
eval_llada_instruct_configs["gpqa_n_shot_gen"]="5|32|32|32|1234|1|0.0"

# ============================================================
# ====================  DREAM CONFIGS  ========================
# ============================================================
#   eval_dream_configs["<dataset>"]="num_fewshot|max_new_tokens|steps|temperature|top_p|seed|mc_num"
# ============================================================

# ---------- Base Generation ----------
eval_dream_base_configs["humaneval_dream"]="0|512|512|0.2|0.95|1234|1"
eval_dream_base_configs["gsm8k_cot"]="8|256|256|0.0|0.95|1234|1"
eval_dream_base_configs["mbpp"]="3|512|512|0.2|0.95|1234|1"
eval_dream_base_configs["minerva_math"]="4|512|512|0.0|0.95|1234|1"
eval_dream_base_configs["bbh"]="3|512|512|0.0|0.95|1234|1"

# ---------- Base Likelihood ----------
eval_dream_base_configs["mmlu"]="5|512|512|0.0|0.95|1234|128"
eval_dream_base_configs["arc_easy"]="0|512|512|0.0|0.95|1234|128"
eval_dream_base_configs["arc_challenge"]="0|512|512|0.0|0.95|1234|128"
eval_dream_base_configs["hellaswag"]="0|512|512|0.0|0.95|1234|128"
eval_dream_base_configs["piqa"]="0|512|512|0.0|0.95|1234|128"
eval_dream_base_configs["gpqa_main_n_shot"]="5|512|512|0.0|0.95|1234|128"
eval_dream_base_configs["winogrande"]="5|512|512|0.0|0.95|1234|128"
eval_dream_base_configs["race"]="0|512|512|0.0|0.95|1234|128"

# ---------- Instruct Generation ----------
eval_dream_instruct_configs["mmlu_generative"]="4|128|128|0.1|0.9|1234|1"
eval_dream_instruct_configs["mmlu_generative_dream"]="4|128|128|0.1|0.9|1234|1"
eval_dream_instruct_configs["mmlu_pro"]="4|128|128|0.1|0.9|1234|1"
eval_dream_instruct_configs["gsm8k_cot"]="0|256|256|0.1|0.9|1234|1"
eval_dream_instruct_configs["minerva_math"]="0|512|512|0.1|0.9|1234|1"
eval_dream_instruct_configs["gpqa_main_n_shot"]="5|128|128|0.0|1.0|1234|1"
eval_dream_instruct_configs["humaneval_instruct"]="0|768|768|0.1|0.9|1234|1"
eval_dream_instruct_configs["mbpp_instruct"]="0|1024|1024|0.1|0.9|1234|1"
eval_dream_instruct_configs["mbpp_instruct_dream"]="0|1024|1024|0.1|0.9|1234|1"
eval_dream_instruct_configs["ifeval"]="0|1280|1280|0.1|0.9|1234|1"

# ============================================================
# ====================  BERT CONFIGS  =========================
# ============================================================
#   eval_bert_configs["<dataset>"]="num_fewshot|max_new_tokens|steps|block_size|seed|mc_num"
# ============================================================

eval_bert_configs["mmlu"]="5|512|512|32|1234|128"
eval_bert_configs["ceval-valid"]="5|1024|1024|32|1234|128"
eval_bert_configs["cmmlu"]="5|1024|1024|32|1234|128"
eval_bert_configs["hellaswag"]="0|1024|1024|1024|1234|128"
eval_bert_configs["winogrande"]="0|128|128|128|1234|128"

eval_bert_configs["gsm8k_bert"]="8|256|256|32|1234|128"
eval_bert_configs["minerva_math"]="4|256|256|32|1234|128"
eval_bert_configs["humaneval"]="0|256|256|32|1234|128"
eval_bert_configs["bbh"]="3|256|256|32|1234|128"


eval_bert_configs["hellaswag_gen"]="0|128|128|128|1234|1"
eval_bert_configs["mmlu_generative"]="0|128|128|128|1234|1"
eval_bert_configs["mmlu_pro"]="0|256|256|256|1234|1"
eval_bert_configs["arc_challenge_chat"]="0|128|128|128|1234|1"

# ============================================================
# ======================  END CONFIGS  ========================
# ============================================================


# ===== Derived variables =====
NUM_NODES=${SLURM_NNODES}
GPUS_PER_NODE=$(echo "${SLURM_JOB_GPUS}" | tr ',' '\n' | wc -l)
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
NODELIST=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
MASTER_ADDR=${NODELIST[0]}
TRAIN_NODES=("${NODELIST[@]}")

echo "============================"
echo "JOB NAME: ${SLURM_JOB_NAME}"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "NUM_NODES: ${NUM_NODES}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "MASTER: ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================"

# ===== Environment =====
export PYTHONBREAKPOINT=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH=.:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1 # For cmmlu dataset
export MASTER_ADDR MASTER_PORT WORLD_SIZE


MODEL_CLASS=${1,,}        # "llada" or "dream"
TASK=${2:-"gsm8k"}        # dataset name
MODEL_NAME=${3}           # model path or name (required)
INSTRUCT=${4:-"False"}    # whether to evaluate instruct model
BATCH_SIZE=${5:-"1"}      # control batchsize
USE_LOG=${6:-"False"}     # optional: enable logging
LIMIT=${7:-"None"}        # optional: limit number of test samples (default None)


if [[ -z "${MODEL_NAME}" ]]; then
  echo "❌ Missing model name/path argument!"
  echo "Usage: sbatch eval_model.sh <model_class> <task> <model_name_or_path> [instruct] [batch_size]"
  exit 1
fi

if [[ "${MODEL_NAME}" == /* ]]; then
  MODEL_PATH="${MODEL_NAME}"
else
  MODEL_PATH="${BASE_MODELS_DIR}/${MODEL_NAME}"
fi

case "${MODEL_CLASS}" in
  llada)
    if [[ "${INSTRUCT,,}" == "true" ]]; then
      CONFIG="${eval_llada_instruct_configs[$TASK]}"
      CONFIG_SET="instruct"
    else
      CONFIG="${eval_llada_base_configs[$TASK]}"
      CONFIG_SET="base"
    fi

    if [[ -z "${CONFIG}" ]]; then
      echo "❌ Unknown task '${TASK}' for LLaDA (${CONFIG_SET} mode)."
      echo "Available tasks (base): ${!eval_llada_base_configs[@]}"
      echo "Available tasks (instruct): ${!eval_llada_instruct_configs[@]}"
      exit 1
    fi

    IFS="|" read -r NUM_FEWSHOT MAX_NEW_TOKENS STEPS BLOCK_SIZE SEED MC_NUM CFG <<< "${CONFIG}"

    MODEL_TYPE="llada"
    SCRIPT_PATH="dllm/pipelines/llada/eval.py"
    MODEL_ARGS="pretrained=${MODEL_PATH},is_check_greedy=False,mc_num=${MC_NUM},max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},block_size=${BLOCK_SIZE},cfg=${CFG}"
    ;;

  dream)
    if [[ "${INSTRUCT,,}" == "true" ]]; then
      CONFIG="${eval_dream_instruct_configs[$TASK]}"
      CONFIG_SET="instruct"
    else
      CONFIG="${eval_dream_base_configs[$TASK]}"
      CONFIG_SET="base"
    fi

    if [[ -z "${CONFIG}" ]]; then
      echo "❌ Unknown task '${TASK}' for Dream (${CONFIG_SET} mode)."
      echo "Available tasks (base): ${!eval_dream_base_configs[@]}"
      echo "Available tasks (instruct): ${!eval_dream_instruct_configs[@]}"
      exit 1
    fi

    IFS="|" read -r NUM_FEWSHOT MAX_NEW_TOKENS STEPS TEMPERATURE TOP_P SEED MC_NUM <<< "${CONFIG}"

    MODEL_TYPE="dream"
    SCRIPT_PATH="dllm/pipelines/dream/eval.py"
    MODEL_ARGS="pretrained=${MODEL_PATH},mc_num=${MC_NUM},max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=true,escape_until=true"
    ;;

  bert)
    CONFIG="${eval_bert_configs[$TASK]}"
    if [[ -z "${CONFIG}" ]]; then
      echo "❌ Unknown task '${TASK}' for BERT."
      echo "Available tasks: ${!eval_bert_configs[@]}"
      exit 1
    fi

    IFS="|" read -r NUM_FEWSHOT MAX_NEW_TOKENS STEPS BLOCK_SIZE SEED MC_NUM <<< "${CONFIG}"

    MODEL_TYPE="bert"
    SCRIPT_PATH="dllm/pipelines/bert/eval.py"
    MODEL_ARGS="pretrained=${MODEL_PATH},is_check_greedy=False,mc_num=${MC_NUM},max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},block_size=${BLOCK_SIZE}"
    ;;

  *)
    echo "❌ Invalid model_class '${MODEL_CLASS}'. Must be 'llada' or 'dream' or 'bert'."
    exit 1
    ;;
esac


[[ "${INSTRUCT}" == "True" ]] && APPLY_CHAT_TEMPLATE_ARG="--apply_chat_template True" || APPLY_CHAT_TEMPLATE_ARG=""
[[ "${LIMIT}" == "None" ]] && LIMIT_ARG="" || LIMIT_ARG="--limit ${LIMIT}"
[[ "${USE_LOG}" == "True" ]] && \
  LOG_ARG="--log_samples --output_path ./.logs/${MODEL_CLASS}_${TASK}_${SLURM_JOB_ID}_samples.json" \
  || LOG_ARG="--output_path ./.logs/${MODEL_CLASS}_${TASK}_${SLURM_JOB_ID}_samples.json"

echo -e "\nLaunching ${MODEL_CLASS} on ${TASK} using ${MODEL_PATH}"
echo "============================"
echo "Few-shot: ${NUM_FEWSHOT}"
echo "Seed: ${SEED}"
echo "Batch size: ${BATCH_SIZE}"
echo "Use chat template: ${USE_CHAT_TEMPLATE}"
echo "============================"

RUN_CMD="accelerate launch \
  --num_processes ${WORLD_SIZE} \
  --num_machines ${NUM_NODES} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --machine_rank ${SLURM_PROCID} \
  ${SCRIPT_PATH} \
  --num_fewshot ${NUM_FEWSHOT} \
  --batch_size ${BATCH_SIZE} \
  --model ${MODEL_TYPE} \
  --model_args \"${MODEL_ARGS}\" \
  --tasks ${TASK} \
  --seed ${SEED} \
  ${LOG_ARG} \
  --confirm_run_unsafe_code \
  ${LIMIT_ARG} \
  ${APPLY_CHAT_TEMPLATE_ARG}"

if [[ "${NUM_NODES}" -eq 1 ]]; then
  echo "Single-node execution"
  eval ${RUN_CMD}
else
  echo "Multi-node execution"
  srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --nodelist="${SLURM_JOB_NODELIST}" ${RUN_CMD}
fi
