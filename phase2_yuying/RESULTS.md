# Phase 2 — Baseline results (Mac)

The numbers Phase 3 must reproduce on CUDA. Debugging history: `/Users/ying/dllm/phase2_yuying/FINDINGS.md`.

## Command
```bash
bash phase2_yuying/run_overfit.sh          # defaults: ROWS=3, STEPS=1000, LoRA r=16, lr 1e-3
python phase2_yuying/merge_lora.py         # -> .models/phase2/overfit/merged
python phase2_yuying/eval_memorization.py .models/phase2/overfit/merged 3
python phase2_yuying/eval_fixed_t.py .models/phase2/overfit/checkpoint-final 3
```
**Code:** dllm commit `a3bb9bb`. It requires the MPS loading fix; without it, training diverges on MPS.

## Environment
| | |
|---|---|
| Machine | Apple M1 Pro, 16 GB unified memory, macOS 26.6.2, device `mps` |
| Software | Python 3.10.20, torch 2.6.0, transformers 4.57.0, peft 0.17.1 |
| Base model | `Qwen/Qwen3-0.6B` → A2D (`dllm/pipelines/a2d/convert.py`) |

## Baseline (run `.models/phase2/overfit_long`)
| Metric | Value |
|---|---|
| Data | `Trelis/tiny-shakespeare[train:3]` → **N = 25** chunks × 64 tokens |
| Trainable params | 10.1M of 606M (1.67%), LoRA r=16, α=32, all-linear |
| Training | 1,000 steps, batch 16, lr 1e-3 constant, fp32, uniform loss weight |
| Wall-clock | 2,422 s (~40 min, ~2.4 s/step) |
| Logged loss, start → end | 3.78 → 0.37 (end/start 0.097); grad norm 1–4 throughout |
| Fixed-mask-ratio accuracy, t = 0.1 / 0.3 / 0.5 / 0.7 / 0.9 | 100% / 99.8% / 99.8% / 99.1% / 70.1% |
| **Memorization** (16-token prompt → 48 tokens) | **94.5%**, with 20 of 25 chunks exact |

**How to read the pass checks:**
- `check_overfit.py` reports PARTIAL even for this run, because its "end loss < 0.3" rule is too strict. See FINDINGS.md.
- The real pass checks are memorization above 90% and fixed-mask-ratio accuracy above 90% at t=0.1.

**For reference, the same flags with `STEPS=300`:** 9.5 min, end loss 0.92, fixed-mask-ratio accuracy 98.8% / 88.2% / 36.2% (t = 0.1 / 0.5 / 0.9), memorization 35.6%. That's not enough training for step 3.

## What changes on a GPU (flags only)
| Mac (Phase 2) | GPU box (Phase 3) | Status |
|---|---|---|
| `--lora True --dtype float32 --bf16 False` | `FULL=1` → full fine-tune, bf16, lr 1e-4 | `FULL=1` is **untested** |
| `python -u examples/a2d/mdlm/pt.py` | Works as-is on 1 GPU; for multi-GPU, `accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes N examples/a2d/mdlm/pt.py` | |
| Loads on CPU, then `.to("mps")` | Loads with `device_map={"": local_process_index}` (unchanged CUDA path) | |
| `[train:3]`, `max_length 64`, 1,000 steps | Full tiny-shakespeare, `max_length 128`, 10 epochs (README recipe) | |
| `loss_weight_type uniform` | `scheduler` (the real objective) | |

**Phase 3, first job:** rerun this exact LoRA command on CUDA and get memorization above 90%. Exact losses will differ because the random numbers differ between devices. Then switch to `FULL=1`.
