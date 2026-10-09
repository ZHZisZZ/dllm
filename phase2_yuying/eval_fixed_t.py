"""Overfit check that ignores training-loss noise: CE at fixed mask ratios.

The logged training loss mixes model quality with the random mask ratio of each
step. This script instead masks every training chunk at fixed ratios
t = 0.1 ... 0.9 (same seed every time) and reports cross-entropy and accuracy on
the masked tokens only.

  memorized model : CE near 0 at t=0.1, rising with t (it uses context)
  context-blind   : about the same CE at every t (only learned token frequencies)

Usage:  python eval_fixed_t.py [adapter_dir | base] [rows] [base_model]
  e.g.  python phase2_yuying/eval_fixed_t.py .models/phase2/overfit/checkpoint-final 3
"""
import sys

import peft
import torch
import torch.nn.functional as F

import dllm

adapter_dir = sys.argv[1] if len(sys.argv) > 1 else ".models/phase2/overfit/checkpoint-final"
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 3
base_dir = sys.argv[3] if len(sys.argv) > 3 else ".models/a2d/Qwen3-0.6B"
SEQ, RATIOS = 64, [0.1, 0.3, 0.5, 0.7, 0.9]

device = "mps" if torch.backends.mps.is_available() else "cpu"
# Load on CPU, then move: loading with device_map={"": "mps"} gives wrong gradients
# (see FINDINGS.md); harmless for this forward-only script, but keep one pattern.
model = dllm.pipelines.a2d.A2DQwen3LMHeadModel.from_pretrained(base_dir, dtype=torch.float32)
if adapter_dir != "base":
    model = peft.PeftModel.from_pretrained(model, adapter_dir)
model = model.to(device).eval()
tokenizer = dllm.utils.get_tokenizer(model_name_or_path=base_dir)

ds = dllm.data.load_pt_dataset(f"Trelis/tiny-shakespeare[train:{rows}]", streaming=False)
grouped = dllm.utils.tokenize_and_group(
    ds["train"][:], tokenizer=tokenizer, text_field="Text",
    seq_length=SEQ, insert_eos=False, drop_tail=True,
)
x = torch.tensor(grouped["input_ids"], device=device)
print(f"{len(x)} training chunks; evaluating {adapter_dir} on {device}\n")

results = []
for t in RATIOS:
    masked = torch.rand(x.shape, generator=torch.Generator().manual_seed(0)) < t
    masked = masked.to(device)
    with torch.no_grad():
        logits = model(input_ids=torch.where(masked, tokenizer.mask_token_id, x)).logits
    ce = F.cross_entropy(logits[masked], x[masked]).item()
    acc = (logits[masked].argmax(-1) == x[masked]).float().mean().item()
    results.append((t, ce, acc))
    print(f"t={t:.1f}  CE {ce:6.2f}  acc {acc:6.1%}")

(_, ce_lo, acc_lo), (_, ce_hi, _) = results[0], results[-1]
print()
if acc_lo > 0.9 and ce_hi > ce_lo + 1.0:
    print("PASS — reconstructs lightly-masked chunks and uses context")
elif ce_hi - ce_lo < 0.5:
    print("FAIL — CE is flat across mask ratios: the model ignores context")
else:
    print("PARTIAL — uses context but has not memorized yet; train longer")
