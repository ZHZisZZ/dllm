"""
LLaDA / MoE / Dream attention mask invariance tests

This script checks:

1) Single-sample padding/mask invariance for multiple base token sequences.
   For each base token sequence, we create 5 variants:

   - "no_padding":    [t1, t2, t3, t4],    mask [1,1,1,1]
   - "left_padding":  [0, t1, t2, t3, t4], mask [0,1,1,1,1]
   - "right_padding": [t1, t2, t3, t4, 0], mask [1,1,1,1,0]
   - "no_mask":       [t1, t2, t3, t4],    mask=None
   - "mask_omitted":  [t1, t2, t3, t4],    attention_mask not passed

   All must produce identical logits on the 4 "real" tokens.

2) Batch vs single consistency:
   Given multiple base sequences, we test:

   (a) No-padding batch:
       stack all base sequences in a batch:
         [[t1_0..3],          # base set 0
          [t1_0..3],          # base set 1
          ...]
       → each row's logits must match the corresponding single-sample "no_padding".

   (b) Padded batch:
       for each base sequence, create two rows:
         right-padded: [t1..t4, 0], mask [1,1,1,1,0]
         left-padded:  [0, t1..t4], mask [0,1,1,1,1]
       batch size = 2 * num_base_sets.
       On the real tokens:
         right-padded rows: positions 0..3
         left-padded rows:  positions 1..4
       → all must match the corresponding single-sample "no_padding".
"""

import gc
from typing import Dict, List

import pytest
import torch
import transformers

import dllm

# Numerical tolerance
ERROR_THRESHOLD = 1e-3

# A list of base token sequences to test.
# You can add more sequences if needed.
BASE_TOKEN_SETS: List[List[int]] = [
    [101, 102, 103, 104],
    [201, 202, 203, 204],
]

# Padding token ID (adjust if your models use a different pad ID)
PAD_TOKEN_ID = 0


def _cuda_cleanup():
    """Free CUDA memory between tests."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            # Not all PyTorch builds expose ipc_collect
            pass


def _get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Get a device for creating input tensors.

    NOTE: with device_map="auto", parameters may be sharded across devices.
    In most HF setups, you can still create inputs on the first parameter's device.
    If your setup differs, adjust this helper accordingly.
    """
    return next(model.parameters()).device


def _build_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Build position_ids so that *real* tokens (mask==1) get contiguous
    positions [0, 1, 2, ...] regardless of left/right padding.

    If attention_mask is None, we treat all positions as real.
    """
    if attention_mask is None:
        mask = torch.ones_like(input_ids, dtype=torch.long)
    else:
        # assume mask is 0/1
        mask = attention_mask.to(dtype=torch.long)

    pos_ids = torch.cumsum(mask, dim=1) - 1  # first real token -> 0
    pos_ids = torch.clamp(pos_ids, min=0)
    return pos_ids.to(dtype=torch.long)


# -------------------------------------------------------------------------
# 1) Single-sample variants over multiple base token sets
# -------------------------------------------------------------------------
def _forward_variants(model, use_position_ids: bool) -> Dict[str, torch.Tensor]:
    """
    For each base token set in BASE_TOKEN_SETS, run 5 variants:

      "no_padding"
      "left_padding"
      "right_padding"
      "no_mask"
      "mask_omitted"

    For each variant we only keep logits on the 4 real token positions.

    Returns:
        dict mapping variant_name -> logits tensor of shape [N, 4, H],
        where N = len(BASE_TOKEN_SETS), in the same order as BASE_TOKEN_SETS.
    """
    device = _get_model_device(model)

    # Accumulators for each variant. We will cat along dim=0 at the end.
    acc = {
        "no_padding": [],
        "left_padding": [],
        "right_padding": [],
        "no_mask": [],
        "mask_omitted": [],
    }

    for base_tokens in BASE_TOKEN_SETS:
        base = torch.tensor([base_tokens], device=device)  # [1,4]
        pad = torch.tensor([[PAD_TOKEN_ID]], device=device)  # [1,1]

        # no_padding
        ids_no_pad = base  # [1,4]
        mask_no_pad = torch.ones_like(ids_no_pad)  # [1,4]

        # left_padding: [0, t1, t2, t3, t4]
        ids_left = torch.cat([pad, base], dim=1)  # [1,5]
        mask_left = torch.cat(
            [torch.zeros_like(pad), torch.ones_like(base)], dim=1
        )  # [1,5]

        # right_padding: [t1, t2, t3, t4, 0]
        ids_right = torch.cat([base, pad], dim=1)  # [1,5]
        mask_right = torch.cat(
            [torch.ones_like(base), torch.zeros_like(pad)], dim=1
        )  # [1,5]

        # no_mask: attention_mask=None
        ids_no_mask = base
        mask_none = None

        # mask_omitted: do not pass attention_mask at all
        ids_omitted = base

        with torch.no_grad():
            # no_padding
            if use_position_ids:
                pos_no_pad = _build_position_ids(ids_no_pad, mask_no_pad)
                out_no_pad = model(
                    input_ids=ids_no_pad,
                    attention_mask=mask_no_pad,
                    position_ids=pos_no_pad,
                ).logits  # [1,4,H]
            else:
                out_no_pad = model(
                    input_ids=ids_no_pad,
                    attention_mask=mask_no_pad,
                ).logits  # [1,4,H]

            # left_padding (slice off pad position)
            if use_position_ids:
                pos_left = _build_position_ids(ids_left, mask_left)
                out_left = model(
                    input_ids=ids_left,
                    attention_mask=mask_left,
                    position_ids=pos_left,
                ).logits[
                    :, 1:
                ]  # [1,4,H]
            else:
                out_left = model(
                    input_ids=ids_left,
                    attention_mask=mask_left,
                ).logits[
                    :, 1:
                ]  # [1,4,H]

            # right_padding (ignore last padded position)
            if use_position_ids:
                pos_right = _build_position_ids(ids_right, mask_right)
                out_right = model(
                    input_ids=ids_right,
                    attention_mask=mask_right,
                    position_ids=pos_right,
                ).logits[
                    :, :-1
                ]  # [1,4,H]
            else:
                out_right = model(
                    input_ids=ids_right,
                    attention_mask=mask_right,
                ).logits[
                    :, :-1
                ]  # [1,4,H]

            # no_mask (attention_mask=None)
            if use_position_ids:
                pos_no_mask = _build_position_ids(ids_no_mask, mask_none)
                out_no_mask = model(
                    input_ids=ids_no_mask,
                    attention_mask=mask_none,
                    position_ids=pos_no_mask,
                ).logits  # [1,4,H]
            else:
                out_no_mask = model(
                    input_ids=ids_no_mask,
                    attention_mask=mask_none,
                ).logits  # [1,4,H]

            # mask_omitted (no attention_mask kwarg)
            if use_position_ids:
                pos_omitted = _build_position_ids(ids_omitted, None)
                out_omitted = model(
                    input_ids=ids_omitted,
                    position_ids=pos_omitted,
                ).logits  # [1,4,H]
            else:
                out_omitted = model(
                    input_ids=ids_omitted,
                ).logits  # [1,4,H]

        acc["no_padding"].append(out_no_pad)
        acc["left_padding"].append(out_left)
        acc["right_padding"].append(out_right)
        acc["no_mask"].append(out_no_mask)
        acc["mask_omitted"].append(out_omitted)

    # Concatenate results for each variant along batch axis.
    outs = {key: torch.cat(tensors, dim=0) for key, tensors in acc.items()}  # [N,4,H]
    return outs


def _assert_invariance(outs: Dict[str, torch.Tensor], tag: str):
    """
    Check that for all base token sets, all variants match "no_padding".
    """
    ref = outs["no_padding"]  # [N,4,H]
    for key in ("left_padding", "right_padding", "no_mask", "mask_omitted"):
        assert torch.allclose(
            ref, outs[key], atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
        ), f"[{tag}] Single-sample mismatch: no_padding vs {key}"


# -------------------------------------------------------------------------
# 2) Batch tests over all base token sets
# -------------------------------------------------------------------------
def _forward_batch_nopad(model, use_position_ids: bool) -> torch.Tensor:
    """
    Batch = stack all base token sets (no padding):

        [
          BASE_TOKEN_SETS[0],  # [t1, t2, t3, t4]
          BASE_TOKEN_SETS[1],
          ...
        ]

    attention_mask = 1 for all positions.

    Returns:
        logits: [N, 4, H] in the same order as BASE_TOKEN_SETS.
    """
    device = _get_model_device(model)

    base_batch = torch.tensor(BASE_TOKEN_SETS, device=device)  # [N,4]
    mask = torch.ones_like(base_batch)  # [N,4]

    with torch.no_grad():
        if use_position_ids:
            pos = _build_position_ids(base_batch, mask)
            logits = model(
                input_ids=base_batch,
                attention_mask=mask,
                position_ids=pos,
            ).logits  # [N,4,H]
        else:
            logits = model(
                input_ids=base_batch,
                attention_mask=mask,
            ).logits  # [N,4,H]

    return logits


def _forward_batch_padded(model, use_position_ids: bool) -> torch.Tensor:
    """
    Padded batch over all base token sets.

    For each base token set base[i] = [t1, t2, t3, t4], we create:

      right-padded row i_r: [t1,t2,t3,t4,0], mask [1,1,1,1,0]
      left-padded  row i_l: [0,t1,t2,t3,t4], mask [0,1,1,1,1]

    We then interleave them in the batch as:

      row 0: base[0] right-padded
      row 1: base[0] left-padded
      row 2: base[1] right-padded
      row 3: base[1] left-padded
      ...

    So the batch size is 2 * N.

    Returns:
        logits: [2N, 5, H]
    """
    device = _get_model_device(model)

    base_batch = torch.tensor(BASE_TOKEN_SETS, device=device)  # [N,4]
    N = base_batch.size(0)

    pad_col = torch.full((N, 1), PAD_TOKEN_ID, device=device)  # [N,1]

    # Right-padded: [t1..t4, 0]
    ids_right = torch.cat([base_batch, pad_col], dim=1)  # [N,5]
    mask_right = torch.cat(
        [torch.ones_like(base_batch), torch.zeros_like(pad_col)], dim=1
    )  # [N,5]

    # Left-padded: [0, t1..t4]
    ids_left = torch.cat([pad_col, base_batch], dim=1)  # [N,5]
    mask_left = torch.cat(
        [torch.zeros_like(pad_col), torch.ones_like(base_batch)], dim=1
    )  # [N,5]

    # Interleave right/left per base:
    # shape [N, 2, 5] -> reshape to [2N, 5]
    ids_stacked = torch.stack([ids_right, ids_left], dim=1)  # [N,2,5]
    mask_stacked = torch.stack([mask_right, mask_left], dim=1)  # [N,2,5]

    batch_ids = ids_stacked.reshape(-1, ids_stacked.size(-1))  # [2N,5]
    batch_mask = mask_stacked.reshape(-1, mask_stacked.size(-1))  # [2N,5]

    with torch.no_grad():
        if use_position_ids:
            pos = _build_position_ids(batch_ids, batch_mask)
            logits = model(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                position_ids=pos,
            ).logits  # [2N,5,H]
        else:
            logits = model(
                input_ids=batch_ids,
                attention_mask=batch_mask,
            ).logits  # [2N,5,H]

    return logits


def _assert_batch_equal_to_single(
    single_no_pad: torch.Tensor,
    batch_nopad: torch.Tensor,
    batch_padded: torch.Tensor,
    tag: str,
):
    """
    Compare batch outputs to single-sample "no_padding" for each base token set.

    Args:
        single_no_pad: [N,4,H] from _forward_variants()["no_padding"]
        batch_nopad:   [N,4,H] from _forward_batch_nopad
        batch_padded:  [2N,5,H] from _forward_batch_padded
    """
    N = single_no_pad.size(0)

    for i in range(N):
        ref = single_no_pad[i : i + 1]  # [1,4,H]
        tokens = BASE_TOKEN_SETS[i]

        # 1) No-padding batch row i
        assert torch.allclose(
            ref,
            batch_nopad[i : i + 1, :, :],
            atol=ERROR_THRESHOLD,
            rtol=ERROR_THRESHOLD,
        ), (
            f"[{tag}] no-pad batch mismatch for base index {i}, " f"tokens={tokens}"
        )

        # 2) Padded batch right-padded row (index 2*i): positions 0..3
        assert torch.allclose(
            ref,
            batch_padded[2 * i : 2 * i + 1, :4, :],
            atol=ERROR_THRESHOLD,
            rtol=ERROR_THRESHOLD,
        ), (
            f"[{tag}] padded batch RIGHT mismatch for base index {i}, "
            f"tokens={tokens} (positions 0..3)"
        )

        # 3) Padded batch left-padded row (index 2*i+1): positions 1..4
        assert torch.allclose(
            ref,
            batch_padded[2 * i + 1 : 2 * i + 2, 1:, :],
            atol=ERROR_THRESHOLD,
            rtol=ERROR_THRESHOLD,
        ), (
            f"[{tag}] padded batch LEFT mismatch for base index {i}, "
            f"tokens={tokens} (positions 1..4)"
        )


# -------------------------------------------------------------------------
# PyTest entry point
# -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model_name_or_path, attn_impl, use_position_ids",
    [
        ("GSAI-ML/LLaDA-8B-Base", None, False),
        ("inclusionAI/LLaDA-MoE-7B-A1B-Base", None, False),
        ("Dream-org/Dream-v0-Base-7B", None, False),
    ],
)
def test_attention_mask_invariance(model_name_or_path, attn_impl, use_position_ids):
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
    model_path = dllm.utils.resolve_with_base_env(model_name_or_path, "BASE_MODELS_DIR")

    # Load model. We assume it's a decoder-style model with .logits.
    if attn_impl is None:
        model = transformers.AutoModel.from_pretrained(
            model_path,
            dtype=torch.float32,
            device_map="auto",
        ).eval()
    else:
        config = transformers.AutoConfig.from_pretrained(
            model_path,
            attn_implementation=attn_impl,
        )
        model = transformers.AutoModel.from_pretrained(
            model_path,
            config=config,
            dtype=torch.float32,
            device_map="auto",
        ).eval()

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
    torch.set_default_device("cuda")
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
def test_a2d_staircase_attention_kvcache_equivalence(
    model_name_or_path, config_cls, model_cls
):
    """
    Verify that staircase attention produces identical logits when run:
        (A) in one full 8-token forward pass
        (B) in two incremental passes (4 tokens → KV cache → 4 tokens)
    """
    from dllm.core.samplers.bd3lm import build_staircase_attention_mask

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------
    # 1. Load model
    # ------------------------------
    model_path = dllm.utils.resolve_with_base_env(model_name_or_path, "BASE_MODELS_DIR")
    config = config_cls.from_pretrained(model_path)
    model = model_cls(config).to(device).eval()

    vocab_size = config.vocab_size
    pad_token_id = 0

    # ------------------------------
    # 2. Create a random 8-token sequence
    # ------------------------------
    block_size = 4
    seq_len = 8

    # Generate random tokens in [1, vocab_size-1]
    x_full = torch.randint(
        low=1,
        high=vocab_size,
        size=(1, seq_len),
        device=device,
        dtype=torch.long,
    )

    # First and second blocks
    x_first = x_full[:, :block_size]  # [1, 4]
    x_second = x_full[:, block_size:seq_len]  # [1, 4]

    # ------------------------------
    # 3. Build staircase mask + positions for the full sequence
    # ------------------------------
    attn_full, pos_full = build_staircase_attention_mask(
        x_full, block_size=block_size, pad_token_id=pad_token_id
    )
    # attn_full: [1, 1, 8, 8]
    # pos_full : [1, 8]

    # Full forward (baseline)
    with torch.no_grad():
        out_full = model(
            input_ids=x_full,
            attention_mask=attn_full,
            position_ids=pos_full,
            use_cache=True,
        )
    logits_full = out_full.logits  # [1, 8, V]

    # ------------------------------
    # 4. Two-step forward with KV cache
    # ------------------------------
    # First block
    attn_first = attn_full[:, :, :block_size, :block_size]  # [1, 1, 4, 4]
    pos_first = pos_full[:, :block_size]  # [1, 4]

    with torch.no_grad():
        out1 = model(
            input_ids=x_first,
            attention_mask=attn_first,
            position_ids=pos_first,
            use_cache=True,
        )
    logits_first = out1.logits
    past_key_values = out1.past_key_values

    # Second block
    attn_second = attn_full[:, :, block_size:seq_len, :seq_len]  # [1, 1, 4, 8]
    pos_second = pos_full[:, block_size:seq_len]  # [1, 4]

    with torch.no_grad():
        out2 = model(
            input_ids=x_second,
            past_key_values=past_key_values,
            attention_mask=attn_second,
            position_ids=pos_second,
            use_cache=True,
        )
    logits_second = out2.logits

    # ------------------------------
    # 5. Compare logits
    # ------------------------------
    diff_first = (logits_full[:, :block_size, :] - logits_first).abs().max().item()
    diff_second = (
        (logits_full[:, block_size:seq_len, :] - logits_second).abs().max().item()
    )

    assert torch.allclose(
        logits_full[:, :block_size, :],
        logits_first,
        atol=ERROR_THRESHOLD,
        rtol=ERROR_THRESHOLD,
    ), f"Mismatch on first block (0–3), max diff={diff_first}"

    assert torch.allclose(
        logits_full[:, block_size:seq_len, :],
        logits_second,
        atol=ERROR_THRESHOLD,
        rtol=ERROR_THRESHOLD,
    ), f"Mismatch on second block (4–7), max diff={diff_second}"

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
