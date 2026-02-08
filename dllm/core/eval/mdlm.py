"""
Generic MDLM/BD3LM-style eval base: _forward_process, get_logits, get_loglikelihood,
_encode_pair, loglikelihood, generate_until. Pipelines inherit and only provide
EvalConfig + _create_sampler + @register_model (and optionally _prepare_prompt_for_generation, _get_sampler_kwargs).

Run: Not runnable directly; use pipeline eval entrypoints (e.g. dllm.pipelines.llada.eval).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from tqdm import tqdm

from dllm.core.eval.base import BaseEvalConfig, BaseEvalHarness
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig


@dataclass
class MDLMEvalConfig(MDLMSamplerConfig, BaseEvalConfig):
    """Common eval config for MDLM-style models (LLaDA, BERT, A2D, etc.)."""

    max_new_tokens: int = 128
    max_length: int = 2048
    steps: int = 128
    block_size: int = 128

    batch_size: int = 32
    mc_num: int = 128
    is_check_greedy: bool = False


class MDLMEvalHarness(BaseEvalHarness):
    """
    Generic MDLM eval: _forward_process, get_logits, get_loglikelihood,
    suffix_greedy_prediction, _encode_pair, loglikelihood, generate_until.
    Subclasses define EvalConfig, _create_sampler(), and optionally
    _prepare_prompt_for_generation(), _get_sampler_kwargs().
    """

    def __init__(
        self,
        config: MDLMEvalConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = MDLMEvalConfig()

        super().__init__(config=config, **kwargs)

        # Pull from config / kwargs (same pattern as existing pipelines)
        batch_size = kwargs.get("batch_size", config.batch_size)
        mc_num = kwargs.get("mc_num", config.mc_num)
        is_check_greedy = kwargs.get("is_check_greedy", config.is_check_greedy)
        cfg_scale = kwargs.get("cfg_scale", getattr(config, "cfg_scale", 0.0))
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        block_size = kwargs.get("block_size", config.block_size)
        max_length = kwargs.get("max_length", config.max_length)
        remasking = kwargs.get("remasking", config.remasking)

        self.mask_id = self.tokenizer.mask_token_id
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.max_new_tokens = int(max_new_tokens)
        self.block_size = int(block_size)
        self.steps = int(steps)
        self.cfg_scale = float(cfg_scale)
        self.remasking = remasking
        self.is_check_greedy = is_check_greedy
        self.mc_num = int(mc_num)
        self.sampling_eps = 0.0

        assert self.mc_num % self.batch_size == 0

    def _forward_process(
        self, batch: torch.Tensor, prompt_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply forward diffusion process by masking a random subset of target tokens."""
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(
            torch.linspace(
                float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device
            )
        ).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat(
            (
                torch.zeros(
                    b, int(prompt_index.sum()), dtype=torch.bool, device=batch.device
                ),
                is_mask,
            ),
            dim=1,
        )

        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        p_mask = (x / target_len).unsqueeze(1).repeat(1, l)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(
        self, batch: torch.Tensor, prompt_index: torch.Tensor
    ) -> torch.Tensor:
        """Plain forward for loglikelihood / suffix_greedy; CFG is handled in the sampler (generate_until)."""
        logits = self.model(batch).logits
        return logits[:, : batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix: torch.Tensor, target: torch.Tensor) -> float:
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
            mask_indices = perturbed_seq == self.mask_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = (
                F.cross_entropy(
                    logits[mask_indices], seq[mask_indices], reduction="none"
                )
                / p_mask[mask_indices]
            )
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(
        self, prefix: torch.Tensor, target: torch.Tensor
    ) -> bool:
        if not self.is_check_greedy:
            return False

        seq = torch.full(
            (1, len(prefix) + len(target)), self.mask_id, device=self.device
        )
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, : len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = seq == self.mask_id
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)
            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(
                dim=-1
            )
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix) :]
        return torch.all(correct).item()

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        out = []
        with torch.no_grad():
            for instance in tqdm(requests, desc="Computing likelihood..."):
                context_enc, continuation_enc = self._encode_pair(*instance.args)
                assert len(context_enc) + len(continuation_enc) <= self.max_length, (
                    f"Context + continuation length exceeds {self.max_length} tokens: "
                    f"{len(context_enc)} + {len(continuation_enc)}"
                )

                context = torch.tensor(
                    context_enc, device=self.device, dtype=torch.long
                )
                continuation = torch.tensor(
                    continuation_enc, device=self.device, dtype=torch.long
                )

                logprob = self.get_loglikelihood(context, continuation)
                isgreedy = self.suffix_greedy_prediction(context, continuation)
                out.append((logprob, isgreedy))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        raise NotImplementedError

    def _create_sampler(self):
        """Return the sampler instance for generate_until. Override for BD3LM etc."""
        return MDLMSampler(model=self.model, tokenizer=self.tokenizer)

    def _get_sampler_kwargs(self) -> dict:
        """Extra kwargs for sampler.sample() in generate_until. Override to add suppress_tokens, etc."""
        return {}

    def _prepare_prompt_for_generation(self, context: str) -> list[torch.Tensor]:
        """Turn context string into list of prompt tensors. Override e.g. for BERT trim [CLS][SEP]."""
        prompt_ids = self.tokenizer(context)["input_ids"]
        return [torch.tensor(prompt_ids, device=self.device, dtype=torch.long)]

    def generate_until(self, requests: list[Instance]) -> list[str]:
        out = []
        sampler = self._create_sampler()
        base_kwargs = {
            "steps": self.steps,
            "max_new_tokens": self.max_new_tokens,
            "block_size": self.block_size,
            "temperature": 0.0,
            "cfg_scale": self.cfg_scale,
            "remasking": self.remasking,
        }
        base_kwargs.update(self._get_sampler_kwargs())

        for instance in tqdm(requests, desc="Generating..."):
            context, gen_kwargs = instance.args  # type: ignore
            prompt = self._prepare_prompt_for_generation(context)
            stop_tokens = gen_kwargs["until"]
            generated_ids = sampler.sample(inputs=prompt, **base_kwargs)
            generated_answer = self.tokenizer.decode(
                generated_ids[0][prompt[0].shape[0] :], skip_special_tokens=False
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
