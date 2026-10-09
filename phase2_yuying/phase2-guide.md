# Phase 2 — Tiny Local Training Loop

**Goal:** prove the MDLM training loop is correct on the Mac before paying for a GPU.
**Deliverable:** one training command that (a) passes an overfit test on ~10 sequences, (b) produces a checkpoint that reproduces its training data through the sampler, and (c) runs on a rented GPU by changing flags only.

Everything below was derived by reading the dllm source (`examples/a2d/mdlm/pt.py`, `dllm/core/trainers/mdlm.py`, `dllm/utils/`). The helper scripts are syntax-checked but **not executed** — run them and expect one or two small fixes.

**Files in this folder**

| File | Purpose |
|---|---|
| `run_overfit.sh` | The training command, Mac-tuned. `FULL=1` switches to full fine-tune for the GPU box. |
| `check_overfit.py` | Reads the training log, prints the loss trajectory and a PASS/FAIL verdict. |
| `merge_lora.py` | Folds the LoRA adapter into the base model so samplers can load it. |
| `eval_memorization.py` | Rebuilds the training chunks, prompts with the first 16 tokens, scores how much of the remaining 48 the model reproduces. |

Copy them into your dllm checkout (e.g. `dllm/phase2/`) and run everything from the repo root, in the `dllm` env.

---

## Step 0 — Know what you are training (30 min, read only)

Open `dllm/core/trainers/mdlm.py` → `compute_loss`. The whole objective is ~40 lines:

1. Sample a timestep `t ∈ [ε, 1)` per sequence.
2. Masking probability `p_mask = 1 − α(t)`; with the default linear schedule that is just `t`.
3. Replace each token with `<|mask|>` independently with probability `p_mask` (prompt / `-100` positions excluded).
4. One forward pass on the noised sequence with **bidirectional** attention.
5. Cross-entropy on masked positions only, multiplied by a loss weight.

Two consequences you need for the overfit test:

- **The loss is noisy by design.** Every step draws a new `t` and a new mask, so a step at `t=0.9` is much harder than one at `t=0.1`. Judge trends over several logging points, not single values.
- **The default weight blows up.** `loss_weight_type="scheduler"` weights by roughly `1/t`, which reaches ~1000× near `t = ε`. That is correct for the ELBO but makes the training loss spiky. For the overfit test use `--loss_weight_type uniform` — a cleaner signal for the same question. Switch back to `scheduler` for real training.

---

## Step 1 — Convert Qwen3-0.6B to an A2D model (~10 min)

A2D needs the AR model re-saved with non-causal attention:

```bash
python dllm/pipelines/a2d/convert.py \
  --model_name_or_path "Qwen/Qwen3-0.6B" \
  --output_dir ".models/a2d/Qwen3-0.6B"
```

Downloads ~1.5GB, runs on CPU (its device helper only knows CUDA, which is fine here). Check that `.models/a2d/Qwen3-0.6B/` contains a config and safetensors.

### Step 1b — Verify the attention actually changed

The README points at `scripts/tests/test_attention.py`, which **doesn't exist** — the tests now live in `scripts/tests/attention/` (add to your friction log). Run only the Qwen3 cases; the Llama case needs a gated HF model:

```bash
pytest scripts/tests/attention/test_invariance_a2d.py -k "Qwen3" -v
```

`test_a2d_fullmask_future_affects_past` is the important one: it changes token 3 and asserts the logits at position 1 change. In an AR model they can't — position 1 can't see the future. If this passes, the conversion really produced bidirectional attention. (It loads `Qwen/Qwen3-0.6B-Base`, another ~1.5GB download.)

---

## Step 2 — The overfit test (~15–40 min on MPS)

```bash
bash phase2/run_overfit.sh
```

What the flags do, and why each differs from the README's GPU command:

| Flag | Value | Why |
|---|---|---|
| `--dataset_args` | `Trelis/tiny-shakespeare[train:3]` | dllm's bracket syntax truncates the split. 3 rows → grouped into 64-token chunks. |
| `--max_length` | `64` | Short chunks = fast steps, easy memorization. |
| `--lora True --r 16` | | Full fine-tune of 0.6B in fp32 needs ~12GB for weights + grads + Adam before activations. On 16GB unified memory that is a crash or heavy swapping. LoRA trains ~1% of parameters. |
| `--dtype float32 --bf16 False` | | bf16 on MPS is version-dependent; fp32 removes one variable. |
| `--loss_weight_type uniform` | | Readable loss curve (Step 0). |
| `--per_device_train_batch_size 16` | | ≥ number of chunks, so every step sees the whole dataset. |
| `--learning_rate 1e-3`, `constant`, no warmup | | Aggressive on purpose; LoRA tolerates a high LR and we *want* overfitting. |
| `--report_to none` | | dllm defaults to `wandb`, which would stop and ask you to log in. |
| `--num_proc 1`, `--dataloader_num_workers 0` | | dllm defaults to 8 processes; macOS multiprocessing tends to hang or crash in `datasets.map`. |
| `--eval_strategy no --save_strategy no` | | Defaults evaluate and save every 10% of training — pure overhead here. A final checkpoint is still written. |

**Before it trains, find this line in the output:** `Num examples = N`. Aim for N ≈ 8–16. If it's much larger, rerun with `ROWS=1`; if smaller, `ROWS=5`.

**While it trains, open Activity Monitor → GPU History.** Unlike `chat.py` in Phase 0, the HF `Trainer` moves the model to `mps` itself, so the GPU should light up. This A/B comparison also pins down Phase 0 finding #8: `dllm/utils/models.py` lines ~40–46 set `device_map` only when CUDA is available, and the chat/sample scripts never move the model afterwards — so inference stays on CPU while training does not. That's your first PR, located.

**Then judge it:**

```bash
python phase2/check_overfit.py .models/phase2/overfit/train.log
```

| Verdict | Meaning | Next |
|---|---|---|
| PASS — end loss < 0.3 and < 10% of start | Loop, data, objective all correct | Step 3 |
| PARTIAL — falling but not collapsed | Probably just under-trained | `STEPS=600`, or `ROWS=1` |
| FAIL — flat or rising | Something is wired wrong | Debug table below — do **not** move on |

Expect a start somewhere in the 3–8 range: the converted model has never seen mask tokens and has just had its attention pattern changed. The exact numbers will differ; the collapse is what matters.

### If it fails

| Symptom | Likely cause | Check |
|---|---|---|
| Loss flat from step 0 | LoRA not attached, or LR effectively 0 | Log should print trainable params ≈ 1–2% of total; confirm `learning_rate` in log lines is 1e-3 |
| Loss `nan` | fp16/bf16 slipped back in, or an MPS kernel issue | Confirm `--bf16 False`; retry with `--use_cpu True` to isolate MPS |
| Loss drops, then plateaus ~1–2 | Too much data for the step budget | `ROWS=1`, or more steps |
| `AssertionError: padding_side == "right"` | Tokenizer loaded from somewhere unexpected | Load from the converted dir, not the hub id |
| `Mismatch between input_ids and labels` | Collator padding labels differently | You changed the collator or data path; revert |
| Hangs at "Mapping dataset" | Multiprocessing on macOS | `--num_proc 1` (already set) |
| MPS out-of-memory | Batch × length too large | `--per_device_train_batch_size 8` |

When debugging, `--use_cpu True` is the fastest way to tell an MPS problem from a code problem: identical run, CPU only, slower but deterministic.

---

## Step 3 — Close the loop: checkpoint → sample → score (~15 min)

A collapsing loss proves the gradient path works. It doesn't prove the checkpoint, the tokenizer save, and the sampler all agree. This step does.

```bash
python phase2/merge_lora.py            # adapter + base → .models/phase2/overfit/merged
python phase2/eval_memorization.py .models/phase2/overfit/merged 3   # rows must match ROWS
```

The eval script rebuilds the exact training chunks, gives the sampler the first 16 tokens of each, lets it denoise the remaining 48, and counts exact token matches. An overfit model should score **> 90%**, and chunk 0's target and output should read near-identical.

It moves the model to MPS explicitly, so it's also a quick test of whether *inference* works on MPS once the device issue is bypassed — useful evidence for the PR.

| Result | Meaning |
|---|---|
| > 90%, text matches | Full pipeline verified: data → tokenize → train → save → load → sample |
| Loss passed but accuracy low | Mismatch between training and inference: tokenizer saved wrong, mask token id differs, or merge failed. Check that `merged/` holds the tokenizer with `<\|mask\|>` |
| Garbled but plausible Shakespeare | Partially memorized; train longer |

---

## Step 4 — Freeze it as "the script you trust"

1. **Commit to your fork.** Fork `ZHZisZZ/dllm` on GitHub, add `phase2/` on a branch, commit. Phase 3 starts with `git clone` on the GPU box, not with remembering flags.
2. **Record a baseline** in `phase2/RESULTS.md`: N examples, start/end loss, steps, wall-clock, memorization accuracy, torch/transformers versions. Phase 3's first job is reproducing these numbers on CUDA.
3. **Document what changes on a GPU — flags only, no code:**

| Mac (Phase 2) | GPU box (Phase 3) |
|---|---|
| `--lora True --dtype float32 --bf16 False` | `FULL=1` → full fine-tune, bf16 |
| `python examples/a2d/mdlm/pt.py` | `accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 examples/a2d/mdlm/pt.py` |
| `[train:3]`, `max_length 64`, 300 steps | Full tiny-shakespeare, `max_length 128`, 10 epochs (README recipe) |
| `loss_weight_type uniform` | `scheduler` (the real objective) |

Rule: the **first** GPU run is this exact overfit test with `FULL=1`. Only once it passes there do you scale data. That catches CUDA-only problems (flash-attn, bf16, DDP) on a 2-minute run instead of a 2-hour one.

---

## Exit criteria

- [ ] `.models/a2d/Qwen3-0.6B` exists; `test_a2d_fullmask_future_affects_past` passes for Qwen3
- [ ] `check_overfit.py` → PASS
- [ ] `eval_memorization.py` → > 90%
- [ ] GPU vs. CPU observation recorded for training (expect GPU) and for `eval_memorization.py`
- [ ] `phase2/` committed to your fork, `RESULTS.md` written

## New friction-log entries

- README references `scripts/tests/test_attention.py`; the file is now `scripts/tests/attention/test_invariance_a2d.py`.
- `wandb` is the default reporter — a fresh run blocks on login.
- `num_proc=8` default in `DataArguments`; fragile on macOS.
- Root cause of Phase 0 #8 located: `get_model()` sets `device_map` only for CUDA; inference scripts never call `.to("mps")`.
