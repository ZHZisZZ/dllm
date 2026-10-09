# Phase 2 — Findings so far (Steps 1–2)

Mac: Apple M1 Pro, 16 GB · torch 2.6.0 · transformers 4.57.0 · branch `fix-bd3lm-mps-eval`

**Bottom line:** Step 2 failed because of a bug in our MPS support, not in the MDLM training loop.
`get_model` loaded weights with `device_map={"": "mps"}` (commit `f6ff4e8`), which gives **wrong gradients** on MPS.
Fixed in `a3bb9bb` (load on CPU, then `.to("mps")`). With the fix, the guide's unchanged `run_overfit.sh` memorizes its data.

---

## Step 1 — Conversion

- `convert.py` copies the tokenizer unchanged (`save_pretrained`). AR and diffusion use the same tokenizer.
  The only addition, `<|mask|>`, is **not saved**: `get_tokenizer` adds it on every load
  (`/Users/ying/dllm/dllm/utils/models.py`, A2D Qwen branch). Hence `mask_token: None` in the converted `tokenizer_config.json`.
- `<|mask|>` gets id **151669**, the first of Qwen3's ~270 padding rows (vocab 151,669 tokens, embedding 151,936 rows).
  That row was never trained (norm 0.36 vs 0.93 for real tokens), and LoRA (`all-linear`) keeps it frozen.
  This turned out **not** to block training.

## Step 1b — Attention test

- `scripts/tests/test_attention.py` doesn't exist; the tests are in `/Users/ying/dllm/scripts/tests/attention/`.
- `test_invariance_a2d.py` hardcoded `torch.set_default_device("cuda")`, so it failed on the Mac. Changed it to fall back cuda → mps → cpu (not committed yet).
- The guide's tests build a **randomly initialized** model from the `Qwen3-0.6B-Base` config. They test the A2D *class*, not your converted checkpoint.
  Added `test_a2d_converted_checkpoint`, which loads `.models/a2d/Qwen3-0.6B`.
  It confirms bidirectional attention: changing token 2 moves position-1 logits by **15.9** (the causal Qwen3-0.6B: **0.0**).

## Step 2 — Overfit test

### Symptom
| | Logged loss | Fixed-mask-ratio CE (t = 0.1 … 0.9) | Accuracy @ t=0.1 |
|---|---|---|---|
| Original run (before fix) | spiky: 7.4 → 12.9 → ~3.7; grad norm up to 765 | **7.2, 7.0, 7.0, 7.1, 7.1**, flat | 8% |
| Same command after fix | smooth: 4.1 → 0.9; grad norm 1–4 | **0.04, 0.12, 0.37, 0.86, 2.99** | **99%** |

Flat CE across mask ratios means the model ignored context and only learned which tokens occur in this text (~7 nats).

### Red herrings (ruled out by experiment)
- **Learning rate:** at 1e-4 the Trainer still diverged; after the fix, 1e-3 is stable.
- **Frozen, untrained mask embedding:** memorization works with it frozen.
- **LoRA rank and `ROWS`:** r=64 made it worse; that was a symptom of the bad gradients, not their cause.
- **`right_shift_logits`:** shifting doesn't help the converted model (CE 17 either way).
- **The earlier MPS NaN fix:** it only touches the BD3LM sampler, which MDLM training never calls.
- **The Trainer itself:** the loss formula, batches, optimizer (AdamW, weight decay 0) and `training_step` all match a plain loop.

### How it was isolated
1. A plain PyTorch loop with the same data, LoRA and learning rate memorized; `pt.py` + `MDLMTrainer` diverged.
2. Swapping in Trainer pieces narrowed it to "same loss, different gradients", even for an identical batch and seed.
3. On CPU both paths gave identical gradients. On MPS the *same input* gave gradient norms of 32, then 541, then 35,000,000.
4. Loading with `.to("mps")` → correct. Loading with `device_map={"": "mps"}` → wrong. That's the only difference.

---

## Why `device_map={"": "mps"}` gave wrong gradients, and why CPU-then-`.to("mps")` fixes it

### What was proven (same fixed batch, gradients compared against CPU)
1. **Forward is correct.** Loss and logits match CPU and are bit-identical across calls (e.g. 17.3023 every call).
2. **The model's state is identical either way.** For both loading paths:
   - every parameter and buffer is bit-identical to a CPU load, including the non-persistent rotary `inv_freq`;
   - the layout is the same: contiguous, storage offset 0, only the tied embedding/`lm_head` shares storage;
   - the autograd flags (`is_leaf`, `requires_grad`, version counter) are the same;
   - SDPA attention is used in both, with no Accelerate hooks.
3. **Activation gradients are correct.** With every weight frozen, the gradients flowing back through all 28 layers match CPU (~1e-4) for both loading paths.
4. **Weight gradients are wrong.** When weights require grad, the `device_map` model's gradients are off on the first backward pass, in every run, by exactly **1.2e-2**. The `.to("mps")` model is off by 3.4e-4, which is ordinary float noise. Later calls vary from run to run: some recover, others grow (0.32 → 79 → 5,100 → 180,000).
   Training feeds each step's gradient into the next step's weights, so the errors compound until training diverges (gradient norm 2.3 million by step 150).
5. **It isn't any single op.** Cross-entropy over the 152k vocab and SDPA (with and without the mask), tested on their own, are correct and repeatable on MPS. No op falls back to CPU.
6. **It isn't memory pressure.** A `.to("mps")` model with 4 GB of extra MPS memory allocated (14.9 GB in use) is still exact.
7. **It depends on what MPS memory the process had already used.** The same `device_map` load gave *correct* gradients when the process had already run another MPS model that was still in the allocator's cache. It gave *wrong* gradients when loaded into a fresh process, or after `torch.mps.empty_cache()`.
8. **Re-copying the weights after a `device_map` load doesn't fix it**, whether by re-cloning the parameters, re-cloning the buffers, or a `.cpu().to("mps")` round trip. Synchronizing after each backward pass doesn't fix it either.
9. **The `.to("mps")` path was correct in every configuration tested**: fresh process, under memory pressure, through the Trainer, and over 600 training steps.

### Interpretation
Nothing in the model differs between the two paths. Same values, same layout, same graph, and the forward pass is correct.
So this is a bug in **PyTorch 2.6's MPS backend**, not in dllm or transformers code. Facts 4, 7 and 8 together point at **Metal memory**: the problem depends on how the MPS allocator has handed out and reused memory blocks, not on the tensors' contents.

The two paths fill the allocator differently:

| Path | What happens on MPS |
|---|---|
| `device_map={"": "mps"}` | transformers loads each tensor straight onto the device, so the parameters are allocated and cast one by one *on MPS* during load |
| CPU load, then `.to("mps")` | the model is built and cast on CPU, then each finished tensor is copied over once |

The first path leaves the allocator in a state where later **weight-gradient** buffers get wrong values: different on the first call (deterministically), and growing or varying after that.

**What's not proven:** the exact faulty kernel or allocator code path. Fact 8 also means the *order* of allocations matters, not just which tensors exist. One more thing ruled out: dllm's `initial_training_setup` disables transformers' `caching_allocator_warmup`, and the bug happens with and without it.

### Why the fix works, and its limits
- `.to("mps")` after a CPU load never puts the allocator in the bad state: correct gradients in every test, and the overfit run memorizes.
- The fix only changes *how* weights reach MPS. CUDA loading is untouched, and so is everything the model or Trainer does afterwards.
- **Caveat:** this is a workaround for an upstream bug we haven't fully root-caused. Worth checking on a newer PyTorch. Repro script:
  `python /Users/ying/dllm/phase2_yuying/repro_mps_device_map.py device_map` (vs `to_mps`); run each in a fresh process.

---

## Notes on the guide's pass criteria

- **The logged loss isn't the CE.** `loss_norm_type=token` divides by *all* tokens, so the logged loss ≈ (fraction masked) × CE. It also swings with each step's random mask ratio.
  Use `/Users/ying/dllm/phase2_yuying/eval_fixed_t.py` instead. It scores fixed mask ratios with a fixed seed, so runs are comparable and a context-blind model is obvious.
- **The `check_overfit.py` threshold (end loss < 0.3) is too strict.** At 90% masking there's almost nothing left to recall the text from, so even a model that has memorized it has a loss floor of about 0.7. The fixed run ended at 0.92 and reported PARTIAL, with 99% accuracy at t=0.1.
- **`ROWS=3` gives 25 chunks, not 8–16.** Each epoch is then two uneven batches (16 + 9).
- **"Start loss 3–8"** only holds because of the normalization above. The converted model's real starting CE is 15–18, worse than random (~12), because bidirectional attention breaks what it learned as an AR model.

## Practical notes (Mac)
- **Long runs:** use `caffeinate -i` and keep the lid open. A closed lid ("Clamshell Sleep") paused a run for about 20 hours.
- **Memory:** swap reached 10 GB of 11 GB at one point, which slows everything. Close other heavy apps.
- **Background scripts:** run with the env's Python directly (`~/miniconda3/envs/dllm/bin/python`). `bash` doesn't load the conda setup from `~/.zshrc`.

## Step 3 — Checkpoint → sample → score

- **Merging is lossless.** The merged model's fixed-mask-ratio scores match the LoRA adapter's exactly, and the merged tokenizer keeps `<|mask|>` = 151669.
- **The 300-step model fails step 3** (35.6%), but not because anything is broken. Step 3 masks the whole **last 48 tokens**, while training masks tokens at random. With 75% masked, one forward pass gets 65.2% right when the masked tokens are scattered, but only 18.7% when they're one contiguous block. The sampler's step-by-step filling raises that to 35.6%, so the sampler works; the model just hasn't memorized whole passages yet.
- **More training fixes it.** Same flags with `STEPS=1000` (40 min):

| | 300 steps | 1,000 steps |
|---|---|---|
| Fixed-mask-ratio accuracy at t = 0.1 / 0.5 / 0.9 | 98.8% / 88.2% / 36.2% | 100% / 99.8% / 70.1% |
| `eval_memorization.py` (16-token prompt → 48 tokens) | 35.6% | **94.5% — PASS** (20 of 25 chunks perfect) |
| `check_overfit.py` | PARTIAL (0.92) | PARTIAL (0.37), so the 0.3 threshold is too strict |

The full pipeline is verified on the Mac: data → tokenize → train → save → merge → load → sample.
Result: `/Users/ying/dllm/.models/phase2/overfit_long/merged`.
One outlier is unexplained: chunk 4 scores 1/48 while every other chunk scores 36/48 or better. It could be correct text shifted by one token; not checked.

## Next
Step 4: freeze the script (use `STEPS=1000`), record the baseline in `RESULTS.md`, and commit `phase2/` to your fork.
