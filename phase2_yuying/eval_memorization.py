"""End-to-end check: can the overfit model reproduce its training chunks?

Rebuilds the exact training chunks, gives the model the first PROMPT tokens of
each, lets the MDLM sampler fill in the rest, and reports token accuracy.

Usage:  python eval_memorization.py [model_dir] [rows]
"""
import sys

import torch

import dllm

model_dir = sys.argv[1] if len(sys.argv) > 1 else ".models/phase2/overfit/merged"
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SEQ, PROMPT = 64, 16
NEW = SEQ - PROMPT  # 48

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = dllm.utils.get_model(model_name_or_path=model_dir, dtype="float32").to(device).eval()
tokenizer = dllm.utils.get_tokenizer(model_name_or_path=model_dir)
sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)

ds = dllm.data.load_pt_dataset(f"Trelis/tiny-shakespeare[train:{rows}]", streaming=False)
grouped = dllm.utils.tokenize_and_group(
    ds["train"][:], tokenizer=tokenizer, text_field="Text",
    seq_length=SEQ, insert_eos=False, drop_tail=True,
)
chunks = grouped["input_ids"]
print(f"{len(chunks)} training chunks; sampling on {device}\n")

config = dllm.core.samplers.MDLMSamplerConfig(
    steps=NEW, max_new_tokens=NEW, block_size=16,
    temperature=0.0, remasking="low_confidence",
)
total_hit = total = 0
for i, chunk in enumerate(chunks):
    prompt, target = chunk[:PROMPT], chunk[PROMPT:]
    out = sampler.sample([prompt], config, return_dict=True)
    gen = out.sequences[0].tolist()[PROMPT:PROMPT + NEW]
    hit = sum(a == b for a, b in zip(gen, target))
    total_hit += hit
    total += len(target)
    print(f"[chunk {i}] {hit}/{len(target)} tokens match")
    if i == 0:
        print("  target:", tokenizer.decode(target).replace("\n", " / "))
        print("  model :", tokenizer.decode(gen).replace("\n", " / "), "\n")

acc = total_hit / max(total, 1)
print(f"\nmemorization accuracy: {acc:.1%}")
print("PASS" if acc > 0.9 else "LOW — the model has not memorized its training set")
