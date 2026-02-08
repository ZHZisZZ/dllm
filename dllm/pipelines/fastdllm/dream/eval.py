"""
accelerate launch \
    --num_processes 4 \
    dllm/pipelines/fastdllm/dream/eval.py \
    --tasks gsm8k_cot \
    --model fastdllm_dream \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,max_new_tokens=256,steps=256,temperature=0.1,top_p=0.9,alg=entropy,dtype=bfloat16"
"""

import logging
from dataclasses import dataclass

import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from tqdm import tqdm

from dllm.pipelines.dream.eval import DreamEvalHarness, eval_logger
from dllm.pipelines.fastdllm.dream import (
    FastdLLMDreamConfig,
    FastdLLMDreamSampler,
    FastdLLMDreamSamplerConfig,
)


@dataclass
class FastdLLMDreamEvalConfig(FastdLLMDreamSamplerConfig):
    top_p: float | None = None
    top_k: float | None = None
    max_new_tokens: int = 128
    max_length: int = 4096
    steps: int = 128
    temperature: float = 0.0
    alg: str = "entropy"

    pretrained: str = ""
    batch_size: int = 1
    device: str = "cuda"
    dtype: str | torch.dtype = "auto"
    add_bos_token: bool = False
    nll_type: str = "mc"
    log_type: str = "ftb"
    mc_num: int = 128
    sampling_eps: float = 1e-3
    escape_until: bool = False
    cfg_scale: float = 0.0  # for DreamEvalHarness compat; unused by FastdLLM
    resolve_pretrained_with_base_env: bool = True

    def get_model_config(self, pretrained: str):
        """Return FastdLLM model config so BaseEvalHarness loads the correct model."""
        return FastdLLMDreamConfig.from_pretrained(pretrained)


@register_model("fastdllm_dream")
class FastdLLMDreamEvalHarness(DreamEvalHarness):
    """Dream eval harness for FastdLLM Dream model; inherits from DreamEvalHarness."""

    def __init__(
        self,
        config: FastdLLMDreamEvalConfig | None = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = FastdLLMDreamEvalConfig()
        super().__init__(config=config, **kwargs)
        use_cache = kwargs.get("use_cache", config.use_cache)
        block_size = kwargs.get("block_size", config.block_size)
        threshold = kwargs.get("threshold", config.threshold)
        self.use_cache = use_cache
        self.block_size = block_size
        self.threshold = threshold
        self.sampler = FastdLLMDreamSampler(model=self.model, tokenizer=self.tokenizer)

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        """Override to pass use_cache, block_size, threshold to FastdLLMDreamSampler."""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        for batch_idx in range(0, len(requests), self.batch_size):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.args for req in batch_requests])

            prompts = list(contexts)
            if self.add_bos_token:
                prompts = [self.tokenizer.bos_token + p for p in prompts]

            prompt_ids = [
                self.tokenizer(p, return_tensors="pt", padding=False)
                .input_ids.squeeze()
                .to(self.device)
                for p in prompts
            ]
            prompt_lens = [len(p_id) for p_id in prompt_ids]

            if max(prompt_lens) > self.max_length - self.max_new_tokens:
                cutoff_len = self.max_length - self.max_new_tokens
                eval_logger.warning(
                    f"Prompt length {max(prompt_lens)} exceeds {cutoff_len}, cutoff on the left side"
                )
                prompt_ids = [p_id[-cutoff_len:] for p_id in prompt_ids]

            generation_ids = self.sampler.sample(
                max_new_tokens=self.max_new_tokens,
                inputs=prompt_ids,
                steps=self.steps,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                alg=self.alg,
                alg_temp=self.alg_temp,
                use_cache=self.use_cache,
                block_size=self.block_size,
                threshold=self.threshold,
                output_history=False,
                return_dict=False,
            )
            cleaned_generation_ids = [
                (
                    seq[seq.ne(self.tokenizer.eos_token_id).float().argmax().long() :]
                    if (seq != self.tokenizer.eos_token_id).any()
                    else seq[-1:]
                )
                for seq in generation_ids
            ]
            truncated_generation_ids = [
                seq[prompt_lens[i] :] for i, seq in enumerate(cleaned_generation_ids)
            ]
            responses = [
                g.removeprefix("<|endoftext|>").split(self.tokenizer.eos_token, 1)[0]
                for g in self.tokenizer.batch_decode(truncated_generation_ids)
            ]

            if not self.escape_until:
                for i, r in enumerate(responses):
                    for s in gen_args[i]["until"]:
                        r = r.split(s)[0]
                    responses[i] = r

            res.extend(responses)
            pbar.update(len(contexts))

        return res


if __name__ == "__main__":
    cli_evaluate()
