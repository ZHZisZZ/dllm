import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# ---------- Configuration (smaller size) ----------
W, H = 480, 210  # lower resolution
TOTAL_DURATION = 3.0
FPS = 15  # lower fps
TEXT = "dLLM"
# TEXT_COLOR = (235, 235, 235)
TEXT_COLOR = (0, 0, 0)
OUTPUT = "logo.gif"
LAST_FRAME_PNG = "logo.png"

DIFFUSION_PORTION = 0.3  # fewer diffusion frames
SEED = 8


# ---------- Auto font size ----------
def load_font_auto_size(text, w, h, target_width_ratio=0.95, target_height_ratio=0.95):
    lo, hi = 10, 2000
    best_font, best_size = None, lo
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=mid
            )
        except OSError:
            font = ImageFont.load_default()
        dummy = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(dummy)
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        width_ok = tw <= w * target_width_ratio
        height_ok = th <= h * target_height_ratio

        if width_ok and height_ok:
            best_font, best_size = font, mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best_font if best_font is not None else font


# ---------- Text rendering ----------
def render_text_mask(w, h, text, font):
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (w - tw) // 2 - bbox[0]
    y = (h - th) // 2 - bbox[1]
    d.text((x, y), text, font=font, fill=255)
    return np.asarray(img, np.float32) / 255.0


# ---------- Initialization ----------
font = load_font_auto_size(TEXT, W, H)
mask = render_text_mask(W, H, TEXT, font)

num_frames = int(TOTAL_DURATION * FPS)
diffusion_frames = max(1, int(num_frames * DIFFUSION_PORTION))
hold_ms = int((TOTAL_DURATION - diffusion_frames / FPS) * 1000)

rng = np.random.default_rng(SEED)
frames = []

# ---------- Diffusion stage ----------
for i in range(diffusion_frames):
    t = i / max(1, diffusion_frames - 1)
    progress = t**0.9
    noise_sigma = (1.0 - progress) ** 2.2

    noise = rng.standard_normal((H, W, 1)).astype(np.float32)
    noise_img = 1.0 - noise_sigma * 0.5 * np.abs(noise)
    np.clip(noise_img, 0.0, 1.0, out=noise_img)

    alpha = progress**2.0
    alpha_map = (mask * alpha).astype(np.float32)[..., None]

    text_rgb = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        text_rgb[..., c] = (mask > 0).astype(np.float32) * (TEXT_COLOR[c] / 255.0)

    frame = (1.0 - alpha_map) * noise_img + alpha_map * text_rgb
    frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    frames.append(Image.fromarray(frame, mode="RGB"))

# ---------- Last frame ----------
final_frame = frames[-1]

# ---------- Save last frame as PNG ----------
final_frame.save(LAST_FRAME_PNG)
print(f"🖼️  Last frame saved as: {LAST_FRAME_PNG}")

# ---------- Quantization (reduce size) ----------
pal_frames = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
pal_final = final_frame.convert("P", palette=Image.ADAPTIVE, colors=64)

# ---------- Save GIF ----------
normal_ms = int(1000 / FPS)
durations = [normal_ms] * len(pal_frames) + [hold_ms]

pal_frames[0].save(
    OUTPUT,
    save_all=True,
    append_images=pal_frames[1:] + [pal_final],
    duration=durations,
    loop=0,
    optimize=True,
)

print(f"✅ GIF saved: {OUTPUT}")
print(
    f"Frames (diffusion only): {len(pal_frames)} at {FPS} FPS, final hold {hold_ms} ms, resolution {W}x{H}"
)
