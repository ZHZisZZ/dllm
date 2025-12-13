# ------------------------------ Visualization (NEW) ------------------------------
# Diffusion-style consecutive output: only show the CURRENT output per frame.
# ------------------ Visualization (sanitized, masks stripped) ------------------
import re
import unicodedata
from typing import Annotated, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


def render_consecutive_trace_gif(
    trace: dict,
    tokenizer,
    out_path: str = "decode_trace.gif",
    font_size: int = 30,
    line_spacing: int = 12,
    frame_ms: int = 250,
    final_ms: int = 5000,  # final clean frame duration (ms)
    max_width: int = 1400,
    max_height: int = 3000,
    margin: int = 32,
    title_color=(80, 80, 80),
    text_color=(0, 0, 0),  # base black
    mask_color=(150, 150, 150),
    sub_nonmask_color=(200, 0, 0),  # persistent red
    ins_color=(0, 0, 200),  # persistent blue
    del_strike_color=(120, 120, 120),
    events_color=(30, 30, 30),
    box_color=(120, 120, 120),
    bg_color=(255, 255, 255),
):
    """
    Persistent coloring keyed by token *instance* (not token id):
      - Inserted tokens -> BLUE across frames (until deleted/substituted again).
      - Substitution nonmask→nonmask -> RED across frames (until deleted/substituted again).
      - Substitution mask→nonmask -> stays BLACK (no extra color).
    Adds a final clean frame (5s) with no events box.
    """
    import unicodedata

    from PIL import Image, ImageDraw, ImageFont

    # ---------- font ----------
    try:
        font = ImageFont.truetype(
            "assets/JetBrainsMono-VariableFont_wght.ttf", font_size
        )
    except Exception:
        print(f"fail to load target font")
        font = ImageFont.load_default()

    # ---------- helpers ----------
    def _sanitize_token(s: str) -> str:
        vis_mask_token = "[m]"
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("Ċ", "\n").replace("▁", " ").replace("Ġ", " ")
        s = s.replace("\t", "    ")
        s = s.replace("\u00a0", " ").replace("\u2007", " ").replace("\u202f", " ")

        # replace mask variants
        if "mdm_mask" in s.lower():
            s = re.sub(r"<[\|]?\s*mdm_mask\s*[\|]?>", "[m]", s, flags=re.IGNORECASE)
            s = s.replace("mdm_mask", "[m]")
        if "mask" in s.lower():
            s = re.sub(r"<[\|]?\s*mask\s*[\|]?>", "[m]", s, flags=re.IGNORECASE)
            s = s.replace("mask", "[m]")

        # replace <|...|> format tokens with bracketed form
        s = re.sub(r"<\|\s*(.*?)\s*\|>", r"[\1]", s)
        return s

    def _tok_str(tok_id: int) -> str:
        try:
            s = tokenizer.decode([int(tok_id)], skip_special_tokens=False)
            s = s if s.strip() else f"<{int(tok_id)}>"
        except Exception:
            s = f"<{int(tok_id)}>"
        return s.replace("\n", "\\n")

    TOKEN_RE = re.compile(r"\s+|\S+")

    def _wrap_text(draw: ImageDraw.ImageDraw, text: str, width_px: int) -> List[str]:
        if text == "":
            return [""]
        lines: List[str] = []
        for para in text.split("\n"):
            tokens = TOKEN_RE.findall(para)
            cur = ""
            for tok in tokens:
                candidate = cur + tok
                if draw.textlength(candidate, font=font) <= width_px:
                    cur = candidate
                else:
                    if cur:
                        lines.append(cur)
                        cur = tok
                        while (
                            draw.textlength(cur, font=font) > width_px and len(cur) > 0
                        ):
                            lo, hi, fit = 1, len(cur), 1
                            while lo <= hi:
                                mid = (lo + hi) // 2
                                if draw.textlength(cur[:mid], font=font) <= width_px:
                                    fit, lo = mid, mid + 1
                                else:
                                    hi = mid - 1
                            lines.append(cur[:fit])
                            cur = cur[fit:]
                    else:
                        t = tok
                        while draw.textlength(t, font=font) > width_px and len(t) > 0:
                            lo, hi, fit = 1, len(t), 1
                            while lo <= hi:
                                mid = (lo + hi) // 2
                                if draw.textlength(t[:mid], font=font) <= width_px:
                                    fit, lo = mid, mid + 1
                                else:
                                    hi = mid - 1
                            lines.append(t[:fit])
                            t = t[fit:]
                        cur = t
            lines.append(cur)
        return lines or [""]

    tmp_img = Image.new("RGB", (10, 10), bg_color)
    tmp_draw = ImageDraw.Draw(tmp_img)
    text_width_budget = max_width - 2 * margin

    # mask detection
    MASK_IDS = set()
    if getattr(tokenizer, "mask_token_id", None) is not None:
        MASK_IDS.add(int(tokenizer.mask_token_id))
    MASK_STRINGS = set()
    mt = getattr(tokenizer, "mask_token", None)
    if mt is not None:
        MASK_STRINGS.add(str(mt))
    MASK_STRINGS.add("<mdm_mask>")

    def _is_mask_token(tok_id: int, tok_str_exact: str) -> bool:
        return (int(tok_id) in MASK_IDS) or (tok_str_exact in MASK_STRINGS)

    def _wrap_tokens_with_index(tokens, deleted_flags):
        lines, cur, cur_w = [], [], 0
        for i, tok in enumerate(tokens):
            t = _sanitize_token(tok)
            parts = t.split("\n")
            for j, seg in enumerate(parts):
                seg_rest = seg
                while seg_rest:
                    w = tmp_draw.textlength(seg_rest, font=font)
                    if cur_w + w <= text_width_budget or not cur:
                        cur.append((seg_rest, i, deleted_flags[i]))
                        cur_w += w
                        seg_rest = ""
                    else:
                        lines.append(cur)
                        cur, cur_w = [], 0
                if j != len(parts) - 1:
                    lines.append(cur)
                    cur, cur_w = [], 0
        if cur:
            lines.append(cur)
        return lines

    def _draw_dashed_rectangle(
        draw, xy, dash=8, gap=6, width=2, outline=(120, 120, 120)
    ):
        x0, y0, x1, y1 = xy
        x = x0
        while x < x1:
            x2 = min(x + dash, x1)
            draw.line([(x, y0), (x2, y0)], fill=outline, width=width)
            draw.line([(x, y1), (x2, y1)], fill=outline, width=width)
            x += dash + gap
        y = y0
        while y < y1:
            y2 = min(y + dash, y1)
            draw.line([(x0, y), (x0, y2)], fill=outline, width=width)
            draw.line([(x1, y), (x1, y2)], fill=outline, width=width)
            y += dash + gap

    def _ops_lines_for_step(st: dict):
        if st is None:
            return ["(no events)"]
        lines = []
        x_before = st["x_before_ids"]
        choose_del = st["choose_del"]
        choose_sub = st["choose_sub"]
        sub_samples = st["sub_samples"]
        ins_samples = st["ins_samples"]
        T = len(x_before)
        for i in range(T):
            if choose_del[i]:
                lines.append(f"DEL@{i}:{_tok_str(int(x_before[i]))}")
            elif choose_sub[i]:
                lines.append(
                    f"SUB@{i}:{_tok_str(int(x_before[i]))}->{_tok_str(int(sub_samples[i]))}"
                )
            if ins_samples[i] is not None:
                lines.append(f"INS@{i}->{i+1}:{_tok_str(int(ins_samples[i]))}")
        if not lines:
            lines.append("(no events)")
        return lines

    # ---- Instance-id machinery ----
    next_instance_id = 0

    def _new_inst():
        nonlocal next_instance_id
        val = next_instance_id
        next_instance_id += 1
        return val

    # Current sequence at the *start* (ids + instance_ids)
    curr_ids = list(trace["init"]["x_ids"])
    curr_inst = [_new_inst() for _ in curr_ids]

    # Persistent color by instance_id: {"blue", "red"}
    color_by_inst = {}

    # ---------- PASS 1: measure required heights per frame ----------
    measurement_payload = []

    for step_idx, st in enumerate([None] + trace["steps"]):
        # build augmented view
        if st is None:
            aug_ids = list(curr_ids)
            deleted_flags = [False] * len(aug_ids)
        else:
            x_before = st["x_before_ids"]
            choose_del = st["choose_del"]
            after_ids = st["x_after_ids"]
            deleted_positions = [i for i, d in enumerate(choose_del) if d]

            aug_ids = list(after_ids)
            deleted_flags = [False] * len(after_ids)
            for i in sorted(deleted_positions, reverse=True):
                aug_ids.insert(i, x_before[i])
                deleted_flags.insert(i, True)

        tokens = tokenizer.convert_ids_to_tokens(aug_ids)
        wrapped_lines = _wrap_tokens_with_index(tokens, deleted_flags)

        # estimate ops lines for this step
        if st:
            ops_text = "  • " + "  • ".join(_ops_lines_for_step(st))
        else:
            ops_text = "(no events)"
        ops_lines = _wrap_text(tmp_draw, ops_text, text_width_budget)

        # compute height needed
        body_h = len(wrapped_lines) * (font_size + line_spacing)
        ops_h = len(ops_lines) * (font_size + line_spacing) + font_size  # + 20
        required_h = margin + (font_size + line_spacing) + body_h + 20

        measurement_payload.append(
            {
                "step_idx": step_idx,
                "st": st,
                "aug_ids": aug_ids,
                "tokens": tokens,
                "deleted_flags": deleted_flags,
                "wrapped_lines": wrapped_lines,
                "ops_lines": ops_lines,
                "required_h": required_h,
            }
        )

    # Measure clean final frame (no events)
    final_text_ids = (
        trace["steps"][-1]["x_after_ids"] if trace["steps"] else trace["init"]["x_ids"]
    )
    final_tokens = tokenizer.convert_ids_to_tokens(final_text_ids)
    wrapped_clean = _wrap_tokens_with_index(final_tokens, [False] * len(final_tokens))
    clean_body_h = len(wrapped_clean) * (font_size + line_spacing)
    clean_required_h = margin + (font_size + line_spacing) + clean_body_h

    # Pick a single uniform canvas height
    max_required_h = max(
        [p["required_h"] for p in measurement_payload] + [clean_required_h]
    )  # + 20
    H = min(max_required_h, max_height)
    W = max_width

    # For each frame we need an augmented view (with deleted placeholders) to draw
    frames = []

    # Iterate steps; for step_idx==0 we still draw "initial state"
    steps_with_initial = [None] + trace["steps"]

    for step_idx, st in enumerate(steps_with_initial):
        if st is None:
            # initial frame: augmented is just current tokens
            aug_ids = list(curr_ids)
            aug_inst = list(curr_inst)
            aug_deleted = [False] * len(aug_ids)
            ops_lines = ["(no events)"]
            title = "initial state"
        else:
            title = f"t = {st['t']:.3f}"
            x_before = list(st["x_before_ids"])
            choose_del = list(st["choose_del"])
            choose_sub = list(st["choose_sub"])
            sub_samples = list(st["sub_samples"])
            ins_samples = list(st["ins_samples"])
            assert (
                len(x_before) == len(curr_ids) == len(curr_inst)
            ), "trace 'x_before' must match current sequence."

            # Build augmented (drawn) and next (state-after) in one pass
            aug_ids, aug_inst, aug_deleted = [], [], []
            next_ids, next_inst = [], []

            for i in range(len(x_before)):
                before_id = int(curr_ids[i])
                before_inst = curr_inst[i]

                if choose_del[i]:
                    # show deleted placeholder (strike-through)
                    aug_ids.append(before_id)
                    aug_inst.append(None)
                    aug_deleted.append(True)
                    # remove from next; also clear any persistent color
                    color_by_inst.pop(before_inst, None)
                else:
                    if choose_sub[i]:
                        after_id = int(sub_samples[i])
                        # in augmented we show the *after* token at same instance
                        aug_ids.append(after_id)
                        aug_inst.append(before_inst)
                        aug_deleted.append(False)
                        next_ids.append(after_id)
                        next_inst.append(before_inst)

                        # update persistence by source type
                        if int(before_id) in MASK_IDS:
                            # mask → nonmask: no extra color (ensure cleared)
                            color_by_inst.pop(before_inst, None)
                        else:
                            # nonmask → nonmask: mark RED
                            color_by_inst[before_inst] = "red"
                    else:
                        # keep
                        aug_ids.append(before_id)
                        aug_inst.append(before_inst)
                        aug_deleted.append(False)
                        next_ids.append(before_id)
                        next_inst.append(before_inst)

                # insertion AFTER position i
                if ins_samples[i] is not None:
                    ins_id = int(ins_samples[i])
                    ins_inst = _new_inst()
                    aug_ids.append(ins_id)
                    aug_inst.append(ins_inst)
                    aug_deleted.append(False)
                    next_ids.append(ins_id)
                    next_inst.append(ins_inst)
                    # mark persistent BLUE for this *instance only*
                    color_by_inst[ins_inst] = "blue"

            # commit next state
            curr_ids, curr_inst = next_ids, next_inst
            ops_text = "  • " + "  • ".join(_ops_lines_for_step(st))
            ops_lines = _wrap_text(tmp_draw, ops_text, text_width_budget)

        # ----- render this frame -----
        tokens = tokenizer.convert_ids_to_tokens(aug_ids)
        wrapped_lines = _wrap_tokens_with_index(tokens, aug_deleted)

        img = Image.new("RGB", (W, H), bg_color)
        draw = ImageDraw.Draw(img)

        y = margin
        draw.text((margin, y), title, fill=title_color, font=font)
        y += font_size + line_spacing

        for line in wrapped_lines:
            x = margin
            for seg_text, tok_idx, is_deleted in line:
                tok_id = int(aug_ids[tok_idx])
                tok_str_exact = tokens[tok_idx]
                inst = aug_inst[tok_idx]

                if is_deleted:
                    # strike deleted — grey masks slightly different if desired
                    strike_color = (
                        mask_color
                        if _is_mask_token(tok_id, tok_str_exact)
                        else del_strike_color
                    )
                    strike = "".join(ch + "\u0336" for ch in seg_text)
                    draw.text((x, y), strike, fill=strike_color, font=font)
                    x += tmp_draw.textlength(strike, font=font)
                else:
                    # choose color by *instance*
                    color = text_color
                    if inst is not None and inst in color_by_inst:
                        color = (
                            ins_color
                            if color_by_inst[inst] == "blue"
                            else sub_nonmask_color
                        )
                    elif _is_mask_token(tok_id, tok_str_exact):
                        color = mask_color
                    draw.text((x, y), seg_text, fill=color, font=font)
                    x += tmp_draw.textlength(seg_text, font=font)
            y += font_size + line_spacing

        # draw events box for all but the extra final-clean frame we'll add later
        # if step_idx != len(steps_with_initial) - 1:
        #     y += 20
        #     x0, y0 = margin, y
        #     x1 = max_width - margin
        #     box_h = len(ops_lines) * (font_size + line_spacing) + font_size + 20
        #     y1 = y0 + box_h
        #     _draw_dashed_rectangle(draw, (x0, y0, x1, y1), outline=box_color)
        #     draw.text((x0 + 10, y0 + 10), "events", fill=events_color, font=font)
        #     yy = y0 + font_size + 20
        #     for l in ops_lines:
        #         draw.text((x0 + 10, yy), l, fill=events_color, font=font)
        #         yy += font_size + line_spacing
        # y += 10
        frames.append(img)

    # ----- extra final clean frame (no events box), 5s -----
    final_ids = list(curr_ids)
    final_inst = list(curr_inst)
    final_tokens = tokenizer.convert_ids_to_tokens(final_ids)

    # wrap without deleted flags
    def _wrap_clean(tokens):
        lines, cur, cur_w = [], [], 0
        for i, tok in enumerate(tokens):
            t = _sanitize_token(tok)
            parts = t.split("\n")
            for j, seg in enumerate(parts):
                seg_rest = seg
                while seg_rest:
                    w = tmp_draw.textlength(seg_rest, font=font)
                    if cur_w + w <= text_width_budget or not cur:
                        cur.append((seg_rest, i))
                        cur_w += w
                        seg_rest = ""
                    else:
                        lines.append(cur)
                        cur, cur_w = [], 0
                if j != len(parts) - 1:
                    lines.append(cur)
                    cur, cur_w = [], 0
        if cur:
            lines.append(cur)
        return lines

    wrapped_clean = _wrap_clean(final_tokens)

    clean_img = Image.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(clean_img)
    draw.text((margin, margin), "final text", fill=title_color, font=font)
    y = margin + font_size + line_spacing
    for line in wrapped_clean:
        x = margin
        for seg_text, tok_idx in line:
            tok_id = int(final_ids[tok_idx])
            tok_str_exact = final_tokens[tok_idx]
            inst = final_inst[tok_idx]
            color = text_color
            if inst in color_by_inst:
                color = (
                    ins_color if color_by_inst[inst] == "blue" else sub_nonmask_color
                )
            elif _is_mask_token(tok_id, tok_str_exact):
                color = mask_color
            draw.text((x, y), seg_text, fill=color, font=font)
            x += tmp_draw.textlength(seg_text, font=font)
        y += font_size + line_spacing
    frames.append(clean_img)

    # save GIF
    durations = [frame_ms] * (len(frames) - 1) + [final_ms]
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=True,
    )
    return out_path
