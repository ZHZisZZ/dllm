"""
Minimal EditFlow τ-leap sampler for EditBase-Dream with diffusion-style visualization.

What changed vs. your original:
- tau_leap_step_minimal returns (x_next, any_edit, step_trace) preserving all intermediates.
- sample_editflow_minimal returns (final_text, trace).
- render_consecutive_trace_gif(trace, tokenizer, ...) draws a GIF where each frame shows
  ONLY the current output (like the Gemini diffusion page shows progressive refinement):
    * SUB tokens in the current frame are orange
    * INS tokens in the current frame are blue
    * KEEP tokens are black
    * If any deletions happened in the step, the title shows ⌫N (red)
"""

# srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --time=03:00:000 python examples/editflow/sample.py --model_name_or_path "models/EditFlow-Dream-Instruct-7B/tulu-3-sft-mixture/checkpoint-final"  --tau 0.02 --mask_length 128 --seed 7070  --prompt "write a romantic story" --make_gif

import math
from dataclasses import dataclass
from typing import Annotated

import torch
import tyro
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from dllm.core.schedulers import BaseKappaScheduler, LinearKappaScheduler

# ------------------------------- Small utilities --------------------------------


def _bernoulli_from_rate(rate: torch.Tensor, tau: float) -> torch.Tensor:
    """First-order τ-leap Bernoulli with p ≈ rate * τ (clamped)."""
    p = (rate.float() * float(tau)).clamp_(0.0, 1.0 - 1e-6)
    return torch.bernoulli(p)


def _sample_from_logits(logits_row: torch.Tensor, temperature: float) -> int:
    """Sample one token id from a 1D logits row with temperature.
    temperature <= 0 -> greedy (argmax).
    """
    if temperature <= 0.0:
        return int(torch.argmax(logits_row).item())
    return int(
        torch.distributions.Categorical(logits=(logits_row / temperature))
        .sample()
        .item()
    )


@dataclass
class GenCfg:
    tau: float = 0.02  # τ step
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1234
    edit_prompt: bool = False  # allow editing inside prompt region?
    temperature: float = 0.7  # token sampling temperature (sub/ins)
    verbose: bool = True  # whether to show intermediate decoding traces
    time_independent: bool = True


# -------------------------------- τ-leap one step --------------------------------


@torch.no_grad()
def tau_leap_step_minimal(
    x: torch.Tensor,  # [T]
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_len: int,  # number of initial prompt tokens (including BOS)
    t: float,
    sched: BaseKappaScheduler,
    cfg: GenCfg,
    prev_out: dict | None = None,  # <-- pass prior step's model outputs
    reuse_prev: bool = False,  # <-- if True, reuse prev_out instead of forward()
) -> tuple[torch.Tensor, bool, dict, dict]:
    """
    Single τ-leap step with deletion/substitution conflict resolution
    and right-insert policy.

    Reuse semantics:
      • If cfg.time_independent == True and reuse_prev == True and prev_out is not None,
        we reuse `prev_out` tensors instead of calling model() again.
      • Otherwise we run a fresh forward().

    Viz-only convention:
      • Any local annotated as _Ann[*, "viz-only"] is used only for human-visible
        tracing / debugging (console logs, GIFs) and does not affect sampling.
      • Such variables are also prefixed with '_' for quick visual scanning.

    Returns:
      x_next, any_edit, _step_trace, out_for_next (the freshly used model outputs)
    """
    device = x.device
    T = x.numel()

    # Decide whether to reuse the previous forward results
    use_reuse = bool(cfg.time_independent and reuse_prev and (prev_out is not None))
    if use_reuse:
        out = prev_out
    else:
        attn = torch.ones(1, T, dtype=torch.long, device=device)
        t_tensor = torch.full((1, 1), float(t), device=device)
        out = model(input_ids=x.unsqueeze(0), attention_mask=attn, t=t_tensor)

    del_rate_h = out["del_rate_hat"]  # [1, T]
    sub_rate_h = out["sub_rate_hat"]  # [1, T]
    ins_rate_h = out["ins_rate_hat"]  # [1, T]
    sub_logits = out["sub_logits"]  # [1, T, V]
    ins_logits = out["ins_logits"]  # [1, T, V]

    # Scale normalized rates to true rates
    tt = torch.tensor([[t]], device=device)
    w = sched.weight(tt)
    del_rate = del_rate_h * w
    sub_rate = sub_rate_h * w
    ins_rate = ins_rate_h * w

    # Clamp prompt_len within current T (robustness)
    prompt_len_clamped = int(max(1, min(prompt_len, T)))

    if not cfg.edit_prompt:
        # Protect the entire prompt span from del/sub
        del_rate[:, :prompt_len_clamped] = 0.0
        sub_rate[:, :prompt_len_clamped] = 0.0
        # Disallow insertions inside the prompt EXCEPT at the last prompt token
        if prompt_len_clamped >= 2:
            ins_rate[:, : prompt_len_clamped - 1] = 0.0

    # Combined "edit" (delete or substitute) event
    comb_rate = (del_rate + sub_rate).squeeze(0)  # [T]
    comb_fire = _bernoulli_from_rate(comb_rate, cfg.tau).bool()  # [T]

    # If an edit fires at i, choose deletion with prob λ_del/(λ_del+λ_sub)
    p_del = (del_rate.squeeze(0) / (comb_rate + 1e-8)).clamp(0, 1)  # [T]
    choose_del = (torch.rand_like(p_del) < p_del) & comb_fire  # [T]
    choose_sub = comb_fire & (~choose_del)  # [T]

    # Insertions (right of token i)
    ins_fire = _bernoulli_from_rate(ins_rate.squeeze(0), cfg.tau).bool()  # [T]

    # Token draws (algorithmic, not viz-only)
    sub_samples: list[int | None] = [
        (
            _sample_from_logits(sub_logits[0, i], cfg.temperature)
            if choose_sub[i]
            else None
        )
        for i in range(T)
    ]
    ins_samples: list[int | None] = [
        _sample_from_logits(ins_logits[0, i], cfg.temperature) if ins_fire[i] else None
        for i in range(T)
    ]

    # Build new sequence left→right (apply insertions to the RIGHT)
    new_ids: list[int] = []

    # --- viz-only per-position labels (for trace/GIF) ---
    _before_ops: Annotated[list[str], "viz-only"] = (
        []
    )  # per 'before' position: DEL/SUB/KEEP
    _after_ops: Annotated[list[str], "viz-only"] = (
        []
    )  # per 'after' token aligned to new_ids: INS/SUB/KEEP

    for i in range(T):
        if choose_del[i]:
            _before_ops.append("DEL")
            # deletion -> no token appended
        elif choose_sub[i]:
            _before_ops.append("SUB")
            new_tok = sub_samples[i]
            new_ids.append(int(new_tok))
            _after_ops.append("SUB")
        else:
            _before_ops.append("KEEP")
            new_ids.append(int(x[i].item()))
            _after_ops.append("KEEP")

        if ins_samples[i] is not None:
            new_ids.append(int(ins_samples[i]))
            _after_ops.append("INS")

    x_next = torch.tensor(new_ids, dtype=torch.long, device=device)
    any_edit = bool(comb_fire.any().item() or ins_fire.any().item())
    # Provide the exact outputs we used this step for the caller to pass forward
    out_for_next = out

    # --- (vis) used only for verbose console trace ---
    if cfg.verbose and (comb_fire.any() or ins_fire.any()):

        def _tok_str(tok_id: int) -> str:  # viz-only helper
            try:
                s = tokenizer.decode([int(tok_id)])
                return s if s.strip() else f"<{int(tok_id)}>"
            except Exception:
                return f"<{int(tok_id)}>"

        _ops_strs: Annotated[list[str], "viz-only"] = []
        for i in range(T):
            if choose_del[i]:
                _ops_strs.append(f"DEL@{i}:{_tok_str(int(x[i]))}")
            elif choose_sub[i]:
                _ops_strs.append(
                    f"SUB@{i}:{_tok_str(int(x[i]))}->{_tok_str(sub_samples[i])}"
                )
            if ins_samples[i] is not None:
                _ops_strs.append(f"INS@{i}->{i+1}:{_tok_str(ins_samples[i])}")
        print("[time]", f"{t:.4f}")
        print("[events]", "; ".join(_ops_strs))
        print("[decode]\n", tokenizer.decode(new_ids, skip_special_tokens=False))
        print()

    # --- (vis) step trace payload (returned; used only for visualization downstream) ---
    _step_trace: Annotated[dict, "viz-only"] = {
        "t": float(t),
        "x_before_ids": [int(i) for i in x.tolist()],
        "x_after_ids": [int(i) for i in new_ids],
        "before_ops": _before_ops,  # viz-only labels
        "after_ops": _after_ops,  # viz-only labels
        # below are algorithmic signals copied for visualization/analysis
        "choose_del": [bool(v) for v in choose_del.tolist()],
        "choose_sub": [bool(v) for v in choose_sub.tolist()],
        "ins_fire": [bool(v) for v in ins_fire.tolist()],
        "sub_samples": [int(s) if s is not None else None for s in sub_samples],
        "ins_samples": [int(s) if s is not None else None for s in ins_samples],
        "prompt_len": prompt_len_clamped,
        "used_reuse": bool(use_reuse),
    }

    return x_next, any_edit, _step_trace, out_for_next


# -------------------------------- top-level sampling -------------------------------


@torch.no_grad()
def sample_editflow_minimal(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    args,
    cfg: GenCfg,
) -> tuple[str, dict]:
    """
    Returns:
        final_text, trace

    Notes on annotations:
      • Any local annotated with Annotated[..., "viz-only"] is only used to build
        the decode trace for visualization (e.g., GIF rendering) and has no effect
        on the actual sampling. Such variables are also prefixed with '_' to make
        this visually obvious in code.
    """
    torch.manual_seed(cfg.seed)

    # If prompt is None, start from BOS alone; otherwise ALWAYS prefix BOS
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        raise ValueError("Tokenizer must have a BOS token for this sampler.")

    prompt = args.prompt
    if prompt is None:
        ids = [bos]  # BOS alone
    else:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        # ids = tokenizer.encode(prompt, add_special_tokens=False)
        # ids = [bos] + enc["input_ids"]  # ALWAYS prefix BOS

    prompt_len = len(ids)

    if args.mask_length:
        if getattr(tokenizer, "mask_token_id", None) is None:
            raise ValueError(
                "Tokenizer must define mask_token_id when --mask_length > 0."
            )
        ids = ids + [tokenizer.mask_token_id] * args.mask_length

    x = torch.tensor(ids, dtype=torch.long, device=model.device)

    sched = LinearKappaScheduler()
    tau = cfg.tau
    steps = math.ceil(1.0 / max(tau, 1e-9))

    _trace: Annotated[dict, "viz-only: full decode trace for GIF/inspection"] = {
        "steps": [],
        "init": {
            "t": 0.0,
            "x_ids": [int(i) for i in x.tolist()],
            "prompt_len": int(prompt_len),
        },
        "end_t": 0.0,
    }

    # Local-only reuse: if previous iteration had no edits, reuse its forward.
    prev_out: dict | None = None
    prev_had_edits = True  # first iteration must run a forward

    t = 0.0
    for _ in range(steps):
        # We can reuse prev_out only if the model is declared time-independent
        # and the previous step had NO edits (sequence unchanged).
        reuse_prev = (
            cfg.time_independent and not prev_had_edits and (prev_out is not None)
        )

        x, edited, _step_trace, prev_out = tau_leap_step_minimal(
            x=x,
            model=model,
            tokenizer=tokenizer,
            prompt_len=prompt_len,
            t=t,
            sched=sched,
            cfg=cfg,
            prev_out=prev_out,
            reuse_prev=reuse_prev,
        )

        _step_trace: Annotated[dict, "viz-only: per-step intermediates for trace"]
        _trace["steps"].append(_step_trace)

        prev_had_edits = edited

        t = min(1.0, t + tau)
        if t >= 1.0 - args.time_epsilon:
            break

    _trace["end_t"] = float(t)

    final_text = tokenizer.decode(x.tolist(), skip_special_tokens=False)
    print("[final]")
    return final_text, _trace


# ---------------------------------------- CLI -------------------------------------


def main():
    @dataclass
    class ScriptArgs:
        # Required (no default)
        model_name_or_path: Annotated[str, "Path or hub id for the model"]
        time_independent: Annotated[
            bool, "Whether model is conditioned on time step"
        ] = True

        prompt: Annotated[str | None, "Text prompt. If None, start from BOS alone."] = (
            None
        )
        # Boolean flag: tyro exposes --edit-prompt / --no-edit-prompt automatically for bools
        edit_prompt: Annotated[
            bool,
            "Allow delete/substitute and insertions in the prompt region (BOS+prompt).",
        ] = False

        # Sampling-related args
        tau: Annotated[float, "τ-leap size"] = 0.01
        time_epsilon: Annotated[
            float, "Match this with the `time_epsilon` arg used in your EditFlowTrainer"
        ] = 1e-3
        mask_length: Annotated[
            int,
            "Number of <mask> tokens appended after the prompt.\n"
            "EditFlow will iteratively substitute, insert, or delete masks to form the output.",
        ] = 128
        temperature: Annotated[float, "Token sampling temperature; 0 for greedy."] = 0.7

        seed: Annotated[int, "Random seed"] = 1234
        verbose: Annotated[bool, "Whether to show intermediate decoding traces"] = True

        # Visualization
        make_gif: Annotated[bool, "Render a decoding trace GIF after sampling."] = False
        gif_path: Annotated[
            str | None, "Output GIF path (default: decode_trace.gif)"
        ] = None
        frame_ms: Annotated[int, "Per-frame duration in ms"] = 120

    args = tyro.cli(ScriptArgs)

    cfg = GenCfg(
        tau=args.tau,
        seed=args.seed,
        edit_prompt=args.edit_prompt,
        temperature=args.temperature,
        verbose=args.verbose,
        time_independent=args.time_independent,
    )

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    final_text, trace = sample_editflow_minimal(model, tokenizer, args, cfg)
    print(final_text)

    if args.make_gif:
        from examples.editflow.viz import render_consecutive_trace_gif

        out = args.gif_path or "decode_trace.gif"
        path = render_consecutive_trace_gif(
            trace,
            tokenizer,
            out_path=out,
            frame_ms=args.frame_ms,
        )
        print(f"[gif saved] {path}")


if __name__ == "__main__":
    main()
