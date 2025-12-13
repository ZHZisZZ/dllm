# LLaDA2.0

> 📄 Tech report: [Scaling Up Diffusion Language Models to 100B](https://github.com/inclusionAI/LLaDA2.0/blob/main/tech_report.pdf) | 💻 Code: [github.com/inclusionAI/LLaDA2.0](https://github.com/inclusionAI/LLaDA2.0/tree/main)

Resources and examples for sampling **LLaDA2.0**.

## Files
```
# pipeline modules relevant with LLaDA2
dllm/pipelines/llada2
├── __init__.py                     # Package initialization
├── models/
│   ├── configuration_llada2_moe.py # LLaDA2-MoE model configuration
│   └── modeling_llada2_moe.py      # LLaDA2-MoE model architecture
└── sampler.py                      # Inference module

# example entry points for inference
examples/llada2
├── chat.py      # Multi-turn chat demo (uses chat template)
├── sample.py    # Single-turn sampling demo
└── README.md    # Documentation (You are here)
```

## Inference
Set `--model_name_or_path` to your checkpoint (e.g., `inclusionAI/LLaDA2.0-mini`).

We support inference for standard sampling:
```shell
python -u examples/llada2/chat.py --model_name_or_path "inclusionAI/LLaDA2.0-mini"
```

We also support interactive multi-turn dialogue with visualization:
```shell
python -u examples/llada2/sample.py --model_name_or_path "inclusionAI/LLaDA2.0-mini"
```
