"""
Generic BD3LM eval base: inherit BaseEvalHarness, implement generate_until with BD3LMSampler.
Loglikelihood is not supported. Pipelines (e.g. a2d) import and register with @register_model.

Run: Not runnable directly; use pipeline eval entrypoints (e.g. dllm.pipelines.a2d.eval).
"""

from dataclasses import dataclass

import torch
from lm_eval.api.instance import Instance
from tqdm import tqdm

from dllm.core.eval.base import BaseEvalConfig, BaseEvalHarness
from dllm.core.samplers import BD3LMSampler, BD3LMSamplerConfig


@dataclass
class BD3LMEvalConfig(BD3LMSamplerConfig, BaseEvalConfig):
    """Eval config for BD3LM: sampler params + base (pretrained, dtype, device)."""

    max_new_tokens: int = 128
    max_length: int = 2048
    steps: int = 128
    block_size: int = 32

    batch_size: int = 32
    mc_num: int = 128
    is_check_greedy: bool = False


class BD3LMEvalHarness(BaseEvalHarness):
    """
    BD3LM eval: BaseEvalHarness + generate_until via BD3LMSampler.
    loglikelihood / loglikelihood_rolling not supported.
    """

    def __init__(
        self,
        config: BD3LMEvalConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = BD3LMEvalConfig()

        super().__init__(config=config, **kwargs)

        batch_size = kwargs.get("batch_size", config.batch_size)
        mc_num = kwargs.get("mc_num", config.mc_num)
        is_check_greedy = kwargs.get("is_check_greedy", config.is_check_greedy)
        cfg_scale = kwargs.get("cfg_scale", getattr(config, "cfg_scale", 0.0))
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        block_size = kwargs.get("block_size", config.block_size)
        max_length = kwargs.get("max_length", config.max_length)
        remasking = kwargs.get("remasking", config.remasking)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)

        self.mask_id = self.tokenizer.mask_token_id
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.max_new_tokens = int(max_new_tokens)
        self.block_size = int(block_size)
        self.steps = int(steps)
        self.cfg_scale = float(cfg_scale)
        self.remasking = remasking
        self.is_check_greedy = is_check_greedy
        self.right_shift_logits = right_shift_logits
        self.mc_num = int(mc_num)

    def generate_until(self, requests: list[Instance]) -> list[str]:
        out = []
        sampler = BD3LMSampler(model=self.model, tokenizer=self.tokenizer)

        for instance in tqdm(requests, desc="Generating..."):
            context, gen_kwargs = instance.args  # type: ignore
            prompt_ids = torch.tensor(
                self.tokenizer(context)["input_ids"],
                device=self.device,
                dtype=torch.long,
            )
            prompt = [prompt_ids]
            stop_tokens = gen_kwargs["until"]
            generated_ids = sampler.sample(
                inputs=prompt,
                steps=self.steps,
                max_new_tokens=self.max_new_tokens,
                block_size=self.block_size,
                temperature=0.0,
                cfg_scale=self.cfg_scale,
                remasking=self.remasking,
                right_shift_logits=self.right_shift_logits,
            )
            # Strip leading pad tokens (BD3LM sampler may return left-padded sequences)
            full_seq = generated_ids[0]
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                while full_seq.numel() > 0 and full_seq[0].item() == pad_id:
                    full_seq = full_seq[1:]
            prompt_len = prompt[0].shape[0]
            gen_ids = full_seq[prompt_len:]
            generated_answer = self.tokenizer.decode(
                gen_ids, skip_special_tokens=False
            )
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(
                generated_answer_ids, skip_special_tokens=True
            )
            out.append(generated_answer)
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("loglikelihood not supported for this model")

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        raise NotImplementedError(
            "loglikelihood_rolling not supported for this model"
        )
