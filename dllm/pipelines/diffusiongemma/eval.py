"""
accelerate launch \
    --num_processes 1 \
    dllm/pipelines/diffusiongemma/eval.py \
    --tasks gsm8k_cot \
    --model diffusiongemma \
    --apply_chat_template \
    --num_fewshot 5 \
    --model_args "pretrained=/mnt/weka/shrd/research/model/diffusiongemma-26B-A4B-it,max_new_tokens=512,steps=48,entropy_bound=0.1,dtype=bfloat16"
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import accelerate
import torch
import transformers
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion


def load_diffusiongemma_sampler():
    @dataclass
    class BaseSamplerOutput:
        sequences: torch.Tensor
        histories: list[torch.Tensor] | None = None

    @dataclass
    class BaseSamplerConfig:
        return_dict: bool = False

    @dataclass
    class BaseSampler:
        model: object
        tokenizer: object
        scheduler: object | None = None

    module_names = ["dllm", "dllm.core", "dllm.core.samplers", "dllm.core.samplers.base"]
    original_modules = {name: sys.modules.get(name) for name in module_names}
    try:
        fake_base = types.ModuleType("dllm.core.samplers.base")
        fake_base.BaseSampler = BaseSampler
        fake_base.BaseSamplerConfig = BaseSamplerConfig
        fake_base.BaseSamplerOutput = BaseSamplerOutput
        sys.modules.setdefault("dllm", types.ModuleType("dllm"))
        sys.modules.setdefault("dllm.core", types.ModuleType("dllm.core"))
        sys.modules.setdefault("dllm.core.samplers", types.ModuleType("dllm.core.samplers"))
        sys.modules["dllm.core.samplers.base"] = fake_base

        sampler_path = Path(__file__).resolve().parent / "sampler.py"
        spec = importlib.util.spec_from_file_location("diffusiongemma_sampler_eval", sampler_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.DiffusionGemmaSampler, module.DiffusionGemmaSamplerConfig
    finally:
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


DiffusionGemmaSampler, DiffusionGemmaSamplerConfig = load_diffusiongemma_sampler()


@dataclass
class DiffusionGemmaEvalSamplerConfig(DiffusionGemmaSamplerConfig):
    """Default sampler config for DiffusionGemma eval."""

    max_new_tokens: int = 256
    steps: int = 48
    entropy_bound: float = 0.1
    entropy_threshold: float = 0.005
    stability_threshold: int = 1
    max_temperature: float = 0.8
    min_temperature: float = 0.4


@dataclass
class DiffusionGemmaEvalConfig:
    """Eval config for DiffusionGemma."""

    pretrained: str = "/mnt/weka/shrd/research/model/diffusiongemma-26B-A4B-it"
    peft_adapter: str | None = None
    metrics_output_path: str | None = None
    device: str = "cuda"
    batch_size: int = 1
    dtype: str = "bfloat16"
    device_map: str | dict[str, int | str] | None = "auto"
    trust_remote_code: bool = False


@register_model("diffusiongemma")
class DiffusionGemmaEvalHarness(LM):
    def __init__(
        self,
        eval_config: DiffusionGemmaEvalConfig | None = None,
        sampler_config: DiffusionGemmaSamplerConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        eval_config = eval_config or DiffusionGemmaEvalConfig()
        sampler_config = sampler_config or DiffusionGemmaEvalSamplerConfig()

        self.pretrained = kwargs.get("pretrained", eval_config.pretrained)
        self.peft_adapter = self._none_if_empty(
            kwargs.get("peft_adapter", kwargs.get("adapter", eval_config.peft_adapter))
        )
        self.metrics_output_path = self._none_if_empty(
            kwargs.get("metrics_output_path", eval_config.metrics_output_path)
        )
        self.batch_size = int(kwargs.get("batch_size", eval_config.batch_size))
        self.requested_device = kwargs.get("device", eval_config.device)
        self.dtype = kwargs.get("dtype", eval_config.dtype)
        self.trust_remote_code = kwargs.get(
            "trust_remote_code", eval_config.trust_remote_code
        )

        self.accelerator = accelerate.Accelerator()
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes

        self.processor = AutoProcessor.from_pretrained(
            self.pretrained,
            trust_remote_code=self.trust_remote_code,
        )
        self.tokenizer = self.processor.tokenizer

        torch_dtype = self._resolve_dtype(self.dtype)
        device_map = self._resolve_device_map(kwargs.get("device_map", eval_config.device_map))
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": self.trust_remote_code,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            self.pretrained,
            **model_kwargs,
        ).eval()
        if self.peft_adapter is not None:
            from peft import PeftModel

            peft_model = PeftModel.from_pretrained(
                self.model,
                self.peft_adapter,
                is_trainable=False,
            ).eval()
            get_base_model = getattr(peft_model, "get_base_model", None)
            self.model = (
                get_base_model().eval()
                if callable(get_base_model)
                else peft_model.eval()
            )
        if device_map is None:
            self.model = self.model.to(self.device)

        self.sampler_config = self._build_config(
            type(sampler_config), sampler_config, kwargs
        )
        self.sampler = DiffusionGemmaSampler(model=self.model, tokenizer=self.tokenizer)
        self._metrics_file = self._rank_metrics_file(self.metrics_output_path)

    @staticmethod
    def _build_config(config_cls, source, kwargs):
        init = {}
        for field in config_cls.__dataclass_fields__.values():
            if field.name in kwargs:
                init[field.name] = kwargs[field.name]
            elif hasattr(source, field.name):
                init[field.name] = getattr(source, field.name)
        return config_cls(**init)

    @staticmethod
    def _none_if_empty(value: Any) -> str | None:
        if value is None:
            return None
        value = str(value)
        if value.lower() in {"", "none", "null"}:
            return None
        return value

    def _rank_metrics_file(self, metrics_output_path: str | None):
        if metrics_output_path is None:
            return None
        path = Path(metrics_output_path)
        if path.suffix:
            path = path.with_name(f"{path.stem}.rank{self.rank}{path.suffix}")
        else:
            path = path / f"metrics.rank{self.rank}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("a", encoding="utf-8", buffering=1)

    @staticmethod
    def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        return getattr(torch, dtype)

    def _resolve_device_map(
        self,
        device_map: str | dict[str, int | str] | None,
    ) -> str | dict[str, int | str] | None:
        if isinstance(device_map, str) and device_map.lower() in {"none", "null"}:
            return None
        if self.accelerator.num_processes > 1 and device_map == "auto":
            return {"": self.accelerator.local_process_index}
        if self.requested_device == "cpu":
            return None
        return device_map

    @property
    def device(self) -> torch.device:
        if self.requested_device == "cuda" and torch.cuda.is_available():
            return torch.device(f"cuda:{self.accelerator.local_process_index}")
        return torch.device(self.requested_device)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    @torch.no_grad()
    def generate_until(
        self,
        requests: list[Instance],
        disable_tqdm: bool = False,
    ) -> list[str]:
        outputs: list[str] = []
        progress = tqdm(
            range(0, len(requests), self.batch_size),
            desc="Generating...",
            disable=disable_tqdm or self.rank != 0,
        )

        for batch_start in progress:
            batch = requests[batch_start : batch_start + self.batch_size]
            contexts, gen_kwargs_list = zip(*[instance.args for instance in batch])
            gen_kwargs = copy.deepcopy(gen_kwargs_list[0])
            stop_sequences = self._normalize_stop_sequences(gen_kwargs.pop("until", None))
            max_gen_toks = gen_kwargs.pop(
                "max_gen_toks", self.sampler_config.max_new_tokens
            )

            prompts = [
                torch.tensor(
                    self.tokenizer(context)["input_ids"],
                    dtype=torch.long,
                    device=self.device,
                )
                for context in contexts
            ]
            sampler_config = copy.deepcopy(self.sampler_config)
            sampler_config.max_new_tokens = int(max_gen_toks)
            sampler_config.return_dict = True
            started_at = time.perf_counter()
            sampler_output = self.sampler.sample(
                prompts,
                sampler_config,
                return_dict=True,
                **gen_kwargs,
            )
            batch_seconds = time.perf_counter() - started_at
            sequences = sampler_output.sequences

            for batch_offset, (sequence, prompt, context, original_gen_kwargs) in enumerate(
                zip(sequences.tolist(), prompts, contexts, gen_kwargs_list)
            ):
                answer = self._decode_generation(sequence, prompt.tolist())
                for stop_sequence in stop_sequences:
                    if stop_sequence and stop_sequence in answer:
                        answer = answer.split(stop_sequence)[0]
                outputs.append(answer)
                self._write_generation_metric(
                    request_index=batch_start + batch_offset,
                    instance=batch[batch_offset],
                    prompt=prompt.tolist(),
                    sequence=sequence,
                    answer=answer,
                    max_gen_toks=max_gen_toks,
                    batch_seconds=batch_seconds,
                    batch_size=len(batch),
                    sampler_output=sampler_output,
                    batch_offset=batch_offset,
                )
                self.cache_hook.add_partial(
                    "generate_until", (context, original_gen_kwargs), answer
                )

            if self.accelerator.num_processes > 1:
                self.accelerator.wait_for_everyone()

        return outputs

    def _write_generation_metric(
        self,
        *,
        request_index: int,
        instance: Instance,
        prompt: list[int],
        sequence: list[int],
        answer: str,
        max_gen_toks: int,
        batch_seconds: float,
        batch_size: int,
        sampler_output: Any,
        batch_offset: int,
    ) -> None:
        if self._metrics_file is None:
            return
        tokens_per_forward = self._tensor_item(
            getattr(sampler_output, "tokens_per_forward", None),
            batch_offset,
        )
        decoder_forward_passes = self._tensor_item(
            getattr(sampler_output, "decoder_forward_passes", None),
            batch_offset,
        )
        new_valid_tokens = self._new_valid_token_count(sequence, len(prompt))
        record = {
            "rank": self.rank,
            "world_size": self.world_size,
            "request_index": request_index,
            "task_name": getattr(instance, "task_name", None),
            "doc_id": getattr(instance, "doc_id", None),
            "prompt_tokens": len(prompt),
            "new_token_slots": max(0, len(sequence) - len(prompt)),
            "new_valid_tokens": new_valid_tokens,
            "answer_chars": len(answer),
            "max_gen_toks": int(max_gen_toks),
            "canvas_length": int(self.model.config.canvas_length),
            "steps": int(self.sampler_config.steps),
            "tokens_per_forward": tokens_per_forward,
            "decoder_forward_passes": decoder_forward_passes,
            "denoising_forwards_per_valid_token": (
                decoder_forward_passes / new_valid_tokens
                if decoder_forward_passes is not None and new_valid_tokens > 0
                else None
            ),
            "batch_size": batch_size,
            "batch_generation_seconds": batch_seconds,
            "seconds_per_sample": batch_seconds / max(batch_size, 1),
            "pretrained": self.pretrained,
            "peft_adapter": self.peft_adapter,
        }
        self._metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _tensor_item(value: Any, index: int) -> float | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().cpu().reshape(-1)[index].item())
        try:
            return float(value[index])
        except (TypeError, IndexError, ValueError):
            return None

    def _new_valid_token_count(self, sequence: list[int], prompt_len: int) -> int:
        new_tokens = sequence[prompt_len:]
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            return len(new_tokens)
        return sum(int(token_id != pad_id) for token_id in new_tokens)

    @staticmethod
    def _normalize_stop_sequences(until: str | list[str] | None) -> list[str]:
        if until is None:
            return []
        if isinstance(until, str):
            return [until]
        return list(until)

    def _decode_generation(self, sequence: list[int], prompt: list[int]) -> str:
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        while sequence and pad_id is not None and sequence[0] == pad_id:
            sequence = sequence[1:]

        start = len(prompt)
        end = len(sequence)
        eos_ids = self._eos_token_ids()
        for index in range(start, len(sequence)):
            if sequence[index] in eos_ids:
                end = index
                break

        return self.tokenizer.decode(sequence[start:end], skip_special_tokens=True)

    def _eos_token_ids(self) -> set[int]:
        token_ids = set()
        for token_id in (
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "eot_token_id", None),
        ):
            if token_id is None:
                continue
            if isinstance(token_id, (list, tuple, set)):
                token_ids.update(int(item) for item in token_id)
            else:
                token_ids.add(int(token_id))
        return token_ids

    def loglikelihood(self, requests):
        raise NotImplementedError(
            "DiffusionGemmaEvalHarness currently supports generation tasks only."
        )

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "DiffusionGemmaEvalHarness currently supports generation tasks only."
        )


if __name__ == "__main__":
    cli_evaluate()
