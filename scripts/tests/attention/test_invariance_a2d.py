"""
A2D attention mask invariance tests.

Run: pytest /mnt/lustrenew/mllm_aligned/dongzhichen/dllm/scripts/tests/attention/test_invariance_a2d.py -v
"""

import gc

import pytest
import torch

import dllm

from .common import (
    BASE_TOKEN_SETS,
    ERROR_THRESHOLD,
    _assert_batch_equal_to_single,
    _assert_invariance,
    _cuda_cleanup,
    _forward_batch_nopad,
    _forward_batch_padded,
    _forward_variants,
)


@pytest.mark.parametrize(
    "model_name_or_path, config_cls, model_cls, attn_impl, use_position_ids",
    [
        # (
        #     "openai-community/gpt2",
        #     dllm.pipelines.a2d.A2DGPT2Config,
        #     dllm.pipelines.a2d.A2DGPT2LMHeadModel,
        #     None,
        #     True
        # ),
        (
            "meta-llama/Llama-3.2-1B",
            dllm.pipelines.a2d.A2DLlamaConfig,
            dllm.pipelines.a2d.A2DLlamaLMHeadModel,
            None,
            False,
        ),
        (
            "Qwen/Qwen2.5-0.5B",
            dllm.pipelines.a2d.A2DQwen2Config,
            dllm.pipelines.a2d.A2DQwen2LMHeadModel,
            None,
            False,
        ),
        (
            "Qwen/Qwen3-0.6B-Base",
            dllm.pipelines.a2d.A2DQwen3Config,
            dllm.pipelines.a2d.A2DQwen3LMHeadModel,
            None,
            False,
        ),
    ],
)
def test_a2d_attention_mask_invariance(
    model_name_or_path,
    config_cls,
    model_cls,
    attn_impl,
    use_position_ids,
):
    """
    For each model:

      1) Single-sample invariance over all base token sets:
           no_padding, left_padding, right_padding,
           no_mask, mask_omitted.

      2) Batch without padding:
           stack all base token sets.

      3) Batch with padding:
           for each base, create right-padded + left-padded rows.

      All logits on the 4 real tokens must match single-sample "no_padding"
      for every base token set.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    torch.set_default_device(device)
    model_path = dllm.utils.resolve_with_base_env(model_name_or_path, "BASE_MODELS_DIR")
    config = config_cls.from_pretrained(model_path, attn_implementation=attn_impl)
    model = model_cls(config)

    # 1) Single-sample variants over all base token sets
    outs_single = _forward_variants(model, use_position_ids=use_position_ids)
    _assert_invariance(outs_single, f"{model_name_or_path}/pos_ids={use_position_ids}")
    single_no_pad = outs_single["no_padding"]  # [N,4,H]

    # 2) Batch (no padding)
    batch_nopad = _forward_batch_nopad(
        model, use_position_ids=use_position_ids
    )  # [N,4,H]

    # 3) Batch (padded, left+right)
    batch_padded = _forward_batch_padded(
        model, use_position_ids=use_position_ids
    )  # [2N,5,H]

    _assert_batch_equal_to_single(
        single_no_pad=single_no_pad,
        batch_nopad=batch_nopad,
        batch_padded=batch_padded,
        tag=f"{model_name_or_path}/pos_ids={use_position_ids}",
    )

    print(
        f"✅ {model_name_or_path} (pos_ids={use_position_ids}) passed: "
        f"mask invariance + batch (no-pad & padded) consistency across "
        f"{len(BASE_TOKEN_SETS)} base token sets within {ERROR_THRESHOLD}."
    )

    del model
    gc.collect()
    _cuda_cleanup()
    torch.set_default_device("cpu")


@pytest.mark.parametrize(
    "model_name_or_path, config_cls, model_cls",
    [
        (
            "meta-llama/Llama-3.2-1B",
            dllm.pipelines.a2d.A2DLlamaConfig,
            dllm.pipelines.a2d.A2DLlamaLMHeadModel,
        ),
        (
            "Qwen/Qwen2.5-0.5B",
            dllm.pipelines.a2d.A2DQwen2Config,
            dllm.pipelines.a2d.A2DQwen2LMHeadModel,
        ),
        (
            "Qwen/Qwen3-0.6B-Base",
            dllm.pipelines.a2d.A2DQwen3Config,
            dllm.pipelines.a2d.A2DQwen3LMHeadModel,
        ),
    ],
)
def test_a2d_fullmask_future_affects_past(model_name_or_path, config_cls, model_cls):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = dllm.utils.resolve_with_base_env(model_name_or_path, "BASE_MODELS_DIR")
    config = config_cls.from_pretrained(model_path)
    model = model_cls(config).to(device).eval()

    a = torch.tensor([[101, 102, 103, 104]], device=device)
    b = torch.tensor([[101, 102, 999, 104]], device=device)

    with torch.no_grad():
        la = model(a).logits
        lb = model(b).logits

    diff = (la[:, 1, :] - lb[:, 1, :]).abs().max().item()
    assert diff > ERROR_THRESHOLD, f"full mask not active, diff={diff}"

    del model
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "converted_path, model_cls",
    [
        (".models/a2d/Qwen3-0.6B", dllm.pipelines.a2d.A2DQwen3LMHeadModel),
    ],
)
def test_a2d_converted_checkpoint(converted_path, model_cls):
    """
    Same checks as above, but on a checkpoint produced by
    dllm/pipelines/a2d/convert.py (real weights instead of random init):
      1) attention-mask / padding / batch invariance
      2) bidirectional attention (future tokens affect past logits)
    """
    import os

    if not os.path.isdir(converted_path):
        pytest.skip(f"converted checkpoint not found: {converted_path}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = (
        model_cls.from_pretrained(converted_path, dtype=torch.float32)
        .to(device)
        .eval()
    )

    # 1) Mask invariance + batch consistency
    outs_single = _forward_variants(model, use_position_ids=False)
    _assert_invariance(outs_single, converted_path)
    _assert_batch_equal_to_single(
        single_no_pad=outs_single["no_padding"],
        batch_nopad=_forward_batch_nopad(model, use_position_ids=False),
        batch_padded=_forward_batch_padded(model, use_position_ids=False),
        tag=converted_path,
    )

    # 2) Bidirectional: changing token 2 must change logits at position 1
    a = torch.tensor([[101, 102, 103, 104]], device=device)
    b = torch.tensor([[101, 102, 999, 104]], device=device)
    with torch.no_grad():
        la = model(a).logits
        lb = model(b).logits
    diff = (la[:, 1, :] - lb[:, 1, :]).abs().max().item()
    assert diff > ERROR_THRESHOLD, f"full mask not active, diff={diff}"

    del model
    gc.collect()
    _cuda_cleanup()
