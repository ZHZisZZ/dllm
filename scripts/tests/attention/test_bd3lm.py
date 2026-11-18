"""
BD3LM attention / KV-cache / concat equivalence tests.

Run from the repo root: python -m pytest scripts/tests/attention/test_bd3lm.py -v
"""

import pytest
import torch

import dllm

from .common import ERROR_THRESHOLD


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
@torch.inference_mode()
def test_a2d_qwen3_bd3lm_boolean_mask_blocks_attention(attn_implementation):
    from dllm.core.samplers.bd3lm import _prepare_for_sampling
    from dllm.pipelines.a2d import A2DQwen3Config, A2DQwen3LMHeadModel

    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        config = A2DQwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            pad_token_id=0,
            attn_implementation=attn_implementation,
        )
        model = A2DQwen3LMHeadModel(config).eval()

    # Keep the same mask while changing a later block or a padding token.
    inputs = torch.tensor([[0, 3, 5, 6], [0, 3, 7, 8], [9, 3, 5, 6]], device="cpu")
    mask, positions = _prepare_for_sampling(inputs[:1], block_size=2, pad_token_id=0)
    logits = model(
        input_ids=inputs,
        attention_mask=mask.expand(inputs.size(0), -1, -1, -1),
        position_ids=positions.expand_as(inputs),
        use_cache=False,
    ).logits

    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits[0, 1], logits[1, 1])
    torch.testing.assert_close(logits[0, 1:], logits[2, 1:])


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
def test_bd3lm_attention_kvcache_equivalence(model_name_or_path, config_cls, model_cls):
    """
    Verify that attention produces identical logits when run:
        (A) in one full 8-token forward pass
        (B) in two incremental passes (4 tokens → KV cache → 4 tokens)
    """
    from dllm.core.samplers.bd3lm import _prepare_for_sampling

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
    # 3. Build mask + positions for the full sequence
    # ------------------------------
    attn_full, pos_full = _prepare_for_sampling(
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
def test_bd3lm_concat_equivalence_when_noised_equals_input(
    model_name_or_path, config_cls, model_cls
):
    """
    Verify that when x_t == x_0 (noised_input == input),
    running the model with BD3LM-style concatenation:

        input_ids = [x_t, x_0], position_ids duplicated, and block-diff attention mask

    produces identical logits on the first half (x_t) as running a normal forward on x_0
    with regular full attention and normal position_ids.

    NOTE: We set block_size == seq_len so x_t tokens attend only within x_t (single block),
          making the first-half computation equivalent to a standard full-attention forward.
    """
    from dllm.core.trainers.bd3lm import _create_bd3lm_attention_mask

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------
    # 1. Load model
    # ------------------------------
    model_path = dllm.utils.resolve_with_base_env(model_name_or_path, "BASE_MODELS_DIR")
    config = config_cls.from_pretrained(model_path)
    model = model_cls(config).to(device).eval()

    vocab_size = config.vocab_size

    # ------------------------------
    # 2. Create random tokens
    # ------------------------------
    seq_len = 32
    block_size = seq_len  # critical for equivalence
    x0 = torch.randint(
        low=1,
        high=vocab_size,
        size=(1, seq_len),
        device=device,
        dtype=torch.long,
    )
    xt = x0.clone()  # noised_input == input

    # ------------------------------
    # 3. Baseline: normal forward on x0 with full attention
    # ------------------------------
    pos = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, L]
    attn_full = torch.ones(1, 1, seq_len, seq_len, device=device, dtype=torch.bool)

    with torch.no_grad():
        out_base = model(
            input_ids=x0,
            attention_mask=attn_full,
            position_ids=pos,
            use_cache=False,
        )
    logits_base = out_base.logits  # [1, L, V]

    # ------------------------------
    # 4. BD3LM-style forward: concat inputs + duplicated pos + specialized mask
    # ------------------------------
    x_cat = torch.cat([xt, x0], dim=1)  # [1, 2L]
    pos_cat = torch.cat([pos, pos], dim=1)  # [1, 2L]

    L2 = 2 * seq_len
    attn_bd = _create_bd3lm_attention_mask(
        b=None,
        h=None,
        q_idx=torch.arange(L2, device=device)[:, None],
        kv_idx=torch.arange(L2, device=device)[None, :],
        block_size=block_size,
        n=seq_len,
    )
    attn_bd = attn_bd.unsqueeze(0).unsqueeze(0)  # [1,1,2L,2L]
    attn_bd = attn_bd.to(device)

    with torch.no_grad():
        out_cat = model(
            input_ids=x_cat,
            attention_mask=attn_bd,
            position_ids=pos_cat,
            use_cache=False,
        )
    logits_cat_first_half = out_cat.logits[:, :seq_len, :]  # [1, L, V]

    # ------------------------------
    # 5. Compare logits (first half only)
    # ------------------------------
    diff = (logits_base - logits_cat_first_half).abs().max().item()

    assert torch.allclose(
        logits_base,
        logits_cat_first_half,
        atol=ERROR_THRESHOLD,
        rtol=ERROR_THRESHOLD,
    ), f"Mismatch on first half logits, max diff={diff}"

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
