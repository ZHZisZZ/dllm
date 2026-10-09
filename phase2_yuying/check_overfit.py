"""Pass/fail verdict for the overfit test, read from the training log.

Usage:  python check_overfit.py .models/phase2/overfit/train.log
"""
import ast
import re
import sys

log_path = sys.argv[1] if len(sys.argv) > 1 else ".models/phase2/overfit/train.log"
losses = []
with open(log_path) as f:
    for line in f:
        m = re.search(r"\{[^{}]*'loss':[^{}]*\}", line)
        if not m:
            continue
        try:
            d = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            continue
        if "loss" in d and "train_runtime" not in d:
            losses.append(float(d["loss"]))

if len(losses) < 6:
    sys.exit(f"Only {len(losses)} loss entries found in {log_path} — did training run?")

first = sum(losses[:2]) / 2
last = sum(losses[-5:]) / 5   # average: MDLM loss is noisy (random mask ratio each step)
ratio = last / first
print(f"logged points : {len(losses)}")
print(f"start loss    : {first:.4f}")
print(f"end loss (avg): {last:.4f}")
print(f"end / start   : {ratio:.3f}")
print("trajectory    :", " ".join(f"{x:.2f}" for x in losses))

if last < 0.3 and ratio < 0.1:
    print("\nPASS — loss collapsed. Loop, data pipeline and objective are wired correctly.")
elif ratio < 0.5:
    print("\nPARTIAL — loss is falling but not collapsed. Try STEPS=600 or a smaller ROWS.")
else:
    print("\nFAIL — loss is not dropping. Do not scale this up; debug first (see guide).")
    sys.exit(1)
