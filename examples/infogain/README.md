# Info-Gain dLLM

> 论文: [Improving Sampling for Masked Diffusion Models via Information Gain](https://arxiv.org/abs/2602.18176)（**ICML 2026** 录用）· 官方实现参考: [Information-Gain-Sampler](https://github.com/yks23/Information-Gain-Sampler)

在 **LLaDA** 与 **Dream** 上提供与 `examples/fastdllm` 对称的推理与评测入口；解码算法来自 Information-Gain-Sampler 的 **候选–打分–提交** 循环，并与本仓库的 `accelerate` / `lm-eval` 流程对齐。

## 目录结构

```
dllm/pipelines/infogain
├── __init__.py
├── info_gain_ops.py          # 与上游一致的熵 / 候选 / 打分工具
├── dream/
│   ├── __init__.py
│   ├── eval.py               # lm-eval: --model infogain_dream
│   ├── sampler.py
│   └── models/
│       ├── configuration_dream.py
│       ├── modeling_dream.py
│       └── __init__.py
└── llada/
    ├── __init__.py
    ├── eval.py               # lm-eval: --model infogain_llada
    ├── sampler.py
    └── models/
        ├── configuration_llada.py
        ├── modeling_llada.py
        └── __init__.py

examples/infogain
├── README.md
├── llada/sample.py
├── llada/eval.sh
├── dream/sample.py
└── dream/eval.sh
```

## 推理

**LLaDA**（`use_cache=None` 走 Info-Gain；`prefix` / `dual` 走 Fast-dLLM 同款 KV 路径）：

```shell
python examples/infogain/llada/sample.py \
  --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
  --use_cache none --threshold 0.8 --candidate_number 8 --position_temperature 0.2 --variant info_gain
```

**Dream**（`use_cache=None` 且 `alg=info_gain`；其它 `alg` 与 Fast-dLLM Dream 行为一致）：

```shell
python examples/infogain/dream/sample.py \
  --model_name_or_path "Dream-org/Dream-v0-Instruct-7B" \
  --use_cache none --alg info_gain --threshold 0.8 --candidate_number 8
```

## 评测

评测前请阅读仓库根目录 [README.md](/README.md) 中的评测环境说明。

```shell
accelerate launch --num_processes 4 \
  dllm/pipelines/infogain/llada/eval.py \
  --tasks gsm8k --num_fewshot 5 --model infogain_llada --apply_chat_template \
  --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,use_cache=none,threshold=0.8,candidate_number=8,max_new_tokens=256,steps=256,block_size=32,suppress_tokens=[],begin_suppress_tokens=[]"

accelerate launch --num_processes 4 \
  dllm/pipelines/infogain/dream/eval.py \
  --tasks gsm8k --num_fewshot 5 --model infogain_dream --apply_chat_template \
  --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,use_cache=none,alg=info_gain,threshold=0.8,candidate_number=8,max_new_tokens=256,steps=256,block_size=32,dtype=bfloat16,add_bos_token=True"
```

或使用脚本：

```shell
bash examples/infogain/llada/eval.sh --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" --instruct True --num_gpu 1
bash examples/infogain/dream/eval.sh --model_name_or_path "Dream-org/Dream-v0-Base-7B" --instruct False --num_gpu 1
```

## 说明

- LLaDA 在 **`use_cache=None`** 时采用 Info-Gain 单步解码；**当前实现要求 batch size 为 1**。多卡评测时请保持每进程 batch 为 1（与常见 `lm-eval` 设置一致）。
- 设置 `use_cache` 为 `prefix` 或 `dual` 时，LLaDA 采样器继承 Fast-dLLM 的 KV 与并行解码逻辑，便于与 Fast-dLLM 基线对比。
- Dream 的 **`alg=info_gain`** 仅在 **`use_cache=None`** 时生效，且 **batch size 须为 1**。
