#!/usr/bin/env bash
# Phase 2 overfit run: MDLM continual-pretraining on a handful of 64-token chunks.
# Run from the dllm repo root, inside the `dllm` conda env.
#   bash /path/to/run_overfit.sh            # LoRA, Mac-sized (default)
#   FULL=1 bash /path/to/run_overfit.sh     # full fine-tune (for the GPU box in Phase 3)
set -euo pipefail

BASE=".models/a2d/Qwen3-0.6B"            # produced by the convert step
OUT=".models/phase2/overfit"
ROWS="${ROWS:-3}"                         # raw dataset rows; 3 rows -> 25 chunks of 64 tokens
STEPS="${STEPS:-1000}"                    # 300 fills random masks but fails step 3 (35.6%);
                                          # 1000 passes step 3 (94.5%), ~40 min on an M1 Pro
mkdir -p "$OUT"

if [[ "${FULL:-0}" == "1" ]]; then
  MODEL_FLAGS=(--lora False --dtype bfloat16 --bf16 True)
  LR=1e-4
else
  MODEL_FLAGS=(--lora True --r 16 --lora_alpha 32 --dtype float32 --bf16 False)
  LR=1e-3
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1

python -u examples/a2d/mdlm/pt.py \
  --model_name_or_path "$BASE" \
  --dataset_args "Trelis/tiny-shakespeare[train:${ROWS}]" \
  --text_field "Text" --insert_eos False --max_length 64 \
  "${MODEL_FLAGS[@]}" \
  --loss_weight_type uniform \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
  --max_steps "$STEPS" --learning_rate "$LR" \
  --lr_scheduler_type constant --warmup_ratio 0 \
  --logging_steps 10 \
  --eval_strategy no --save_strategy no \
  --report_to none --num_proc 1 --group_by_length False \
  --dataloader_num_workers 0 --dataloader_pin_memory False \
  --output_dir "$OUT" 2>&1 | tee "$OUT/train.log"
