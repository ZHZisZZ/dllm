# Info-Gain dLLM

> Paper: [Improving Sampling for Masked Diffusion Models via Information Gain](https://arxiv.org/abs/2602.18176) (**ICML 2026**) | Reference implementation: [Information-Gain-Sampler](https://github.com/yks23/Information-Gain-Sampler)

Resources and examples for inferencing and evaluating **LLaDA** and **Dream** with the **Information Gain** sampler. Layout mirrors [`examples/fastdllm`](/examples/fastdllm): the decode loop follows the upstream **propose–score–commit** pattern, wired to this repo’s **`accelerate` / `lm-eval`** harness.

## Table of Contents
- [Files](#files)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Notes](#notes)

## Files

```
# Pipeline modules for Info-Gain
dllm/pipelines/infogain
├── __init__.py
├── info_gain_ops.py                # Entropy / candidates / scoring (aligned with upstream)
├── dream/
│   ├── __init__.py
│   ├── models/
│   │   ├── configuration_dream.py  # Info-Gain Dream model configuration
│   │   └── modeling_dream.py       # Info-Gain Dream model architecture
│   ├── sampler.py                  # Info-Gain Dream inference module
│   └── eval.py                     # lm-eval harness: --model infogain_dream
└── llada/
    ├── __init__.py
    ├── models/
    │   ├── configuration_llada.py  # Info-Gain LLaDA model configuration
    │   └── modeling_llada.py       # Info-Gain LLaDA model architecture
    ├── sampler.py                  # Info-Gain LLaDA inference module
    └── eval.py                     # lm-eval harness: --model infogain_llada

# Example entry points for inference and evaluation
examples/infogain
├── README.md                       # Documentation (you are here)
├── dream/
│   ├── sample.py                   # Info-Gain Dream inference example
│   └── eval.sh                     # Info-Gain Dream evaluation script (baseline + Info-Gain)
└── llada/
    ├── sample.py                   # Info-Gain LLaDA inference example
    └── eval.sh                     # Info-Gain LLaDA evaluation script (baseline + Info-Gain + prefix cache)
```

## Inference

> With **`use_cache=none`**, LLaDA uses Info-Gain per-step decoding. With **`use_cache=prefix`** or **`dual`**, the sampler reuses the same KV / parallel path as Fast-dLLM for apples-to-apples baselines.

Sampling with the Info-Gain LLaDA sampler (`use_cache none`, confidence threshold, candidate count, position temperature, `variant=info_gain`):

```shell
# --use_cache none enables Info-Gain; --threshold / --candidate_number / --position_temperature tune the sampler
python examples/infogain/llada/sample.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --use_cache none --threshold 0.8 --candidate_number 8 --position_temperature 0.2 --variant info_gain
```

The Info-Gain objective is weighted as
`J = -cost_weight * C - future_weight * H_next`. By default,
`cost_weight=1`, `future_weight=1`, and `C` is the sum of entropies over
the decoded action. Set `info_gain_cost_reduction=mean` to use
**entropy\***, i.e. the average entropy of the token(s) decoded by the
current action. The mean reduction can be stronger than the default sum
reduction in some settings, especially when comparing candidate actions with
different decoded-token counts or under very small decoding-step budgets:

```shell
python examples/infogain/llada/sample.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --use_cache none --threshold 0.8 --candidate_number 8 --position_temperature 0.2 \
    --variant info_gain --info_gain_cost_reduction mean \
    --info_gain_cost_weight 1.5 --info_gain_future_weight 1.0
```

Sampling with the Info-Gain Dream sampler (`use_cache none`, `alg=info_gain`; other `alg` values match Fast-dLLM Dream):

```shell
# --use_cache none and --alg info_gain enable Information-Gain decoding; --threshold / --candidate_number match the paper-style loop
python examples/infogain/dream/sample.py \
    --model_name_or_path "Dream-org/Dream-v0-Instruct-7B" \
    --use_cache none --alg info_gain --threshold 0.8 --candidate_number 8
```

Dream accepts the same Info-Gain weighting controls through
`info_gain_cost_reduction`, `info_gain_cost_weight`, and
`info_gain_future_weight`.

## Correctness smoke tests

The repository includes GPU smoke tests for sampler correctness:

```shell
# Toy-model GPU test: verifies Info-Gain no-cache decoding and candidate scoring.
python scripts/tests/test_infogain_gpu_smoke.py

# Toy-model GPU test: verifies Info-Gain prefix/dual cache paths match Fast-dLLM
# for both LLaDA and Dream, including past_key_values and replace_position use.
python scripts/tests/test_infogain_fastdllm_cache_gpu.py
```

For real checkpoint inference, use `test_infogain_real_inference.py`. It loads an
actual LLaDA or Dream checkpoint, decodes a simple arithmetic prompt, and fails if
the decoded answer is empty, still contains a mask token, or does not match the
expected answer regex.

```shell
python scripts/tests/test_infogain_real_inference.py \
    --pipeline llada \
    --model_name_or_path /path/to/LLaDA-8B-Instruct \
    --modes none \
    --expected_regex '\b(96|ninety[- ]?six)\b'

python scripts/tests/test_infogain_real_inference.py \
    --pipeline dream \
    --model_name_or_path /path/to/Dream-v0-Instruct-7B \
    --modes none \
    --expected_regex '\b(96|ninety[- ]?six)\b'
```

Add `prefix dual` to `--modes` when validating the Fast-dLLM cache branches with
the same real checkpoint:

```shell
python scripts/tests/test_infogain_real_inference.py \
    --pipeline llada \
    --model_name_or_path /path/to/LLaDA-8B-Instruct \
    --modes none prefix dual
```

## Evaluation

> Read [(optional) Evaluation setup](/README.md#optional-evaluation-setup) before running evaluation.

For example, to evaluate [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) or [Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) on [GSM8K](https://huggingface.co/datasets/openai/gsm8k) with 4 GPUs, run:

```shell
# Pass Info-Gain options via model_args (use_cache=none, threshold, candidate_number, position_temperature, variant, weighting, etc.).
accelerate launch --num_processes 4 \
    dllm/pipelines/infogain/llada/eval.py \
    --tasks "gsm8k" \
    --num_fewshot 5 \
    --model "infogain_llada" \
    --apply_chat_template \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,use_cache=none,threshold=0.8,candidate_number=8,position_temperature=0.2,variant=info_gain,max_new_tokens=256,steps=256,block_size=32,suppress_tokens=[],begin_suppress_tokens=[]"

accelerate launch --num_processes 4 \
    dllm/pipelines/infogain/dream/eval.py \
    --tasks "gsm8k" \
    --num_fewshot 5 \
    --model "infogain_dream" \
    --apply_chat_template \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,use_cache=none,alg=info_gain,threshold=0.8,candidate_number=8,position_temperature=0.2,info_gain_variant=info_gain,temperature=0.0,top_p=0.9,max_new_tokens=256,steps=256,block_size=32,dtype=bfloat16,add_bos_token=True"
```

To run the bundled scripts (they launch **standard** `llada` / `dream` evals plus **Info-Gain**; the LLaDA script also runs **Info-Gain + Fast-dLLM prefix cache** on GSM8K), use:

```shell
bash examples/infogain/llada/eval.sh --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" --instruct True --num_gpu 1
bash examples/infogain/dream/eval.sh --model_name_or_path "Dream-org/Dream-v0-Base-7B" --instruct False --num_gpu 1
```

Optional flags for `eval.sh`: `--max_new_tokens`, `--threshold`, `--candidate_number` (LLaDA and Dream); `--block_size` (Dream only).

## Notes

- **Batch size**: Info-Gain no-cache decoding supports batched inputs for both LLaDA (`use_cache=none`) and Dream (`alg=info_gain` with `use_cache=none`). The smoke tests also cover the Fast-dLLM-compatible prefix / dual cache paths.
- **Objective weighting**: `info_gain_cost_reduction=sum` is the default and matches the original immediate cost. `info_gain_cost_reduction=mean` enables **entropy\***, the average entropy of the decoded action tokens. `info_gain_cost_weight` and `info_gain_future_weight` control the immediate-cost and future-uncertainty terms.
- **LLaDA**: `use_cache=prefix` or `dual` uses the Fast-dLLM-style KV and parallel decode path inside the Info-Gain pipeline module (not the inner Info-Gain objective), useful for comparing against Fast-dLLM.
- **Dream**: `alg=info_gain` is only active when **`use_cache=none`**; other settings follow the Dream sampler’s Fast-dLLM-compatible branches.
