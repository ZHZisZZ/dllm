"""Merge the LoRA adapter into the base A2D model so the samplers can load it.

Usage:  python merge_lora.py [base_dir] [adapter_dir] [out_dir]
"""
import sys

import peft
import transformers

import dllm

base = sys.argv[1] if len(sys.argv) > 1 else ".models/a2d/Qwen3-0.6B"
adapter = sys.argv[2] if len(sys.argv) > 2 else ".models/phase2/overfit/checkpoint-final"
out = sys.argv[3] if len(sys.argv) > 3 else ".models/phase2/overfit/merged"

model = dllm.utils.get_model(model_name_or_path=base, dtype="float32")
model = peft.PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out)
transformers.AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
print(f"merged model saved to {out}")
