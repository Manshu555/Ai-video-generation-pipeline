"""
Generates scene illustrations.
Priority: Local SDXL (RTX 4060 GPU) -> Pollinations.ai -> PIL styled placeholder
fal.ai removed — balance exhausted.
"""
import time
import urllib.parse
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from config import CHARACTER_DESCRIPTION, VIDEO_WIDTH, VIDEO_HEIGHT, SDXL_CACHE_DIR, PREFER_FAL_IMAGE

# ── Prompt templates ──────────────────────────────────────────────────────────
# Kept concise: SD 1.5's CLIP truncates at 77 tokens, so trailing words are lost.
# "no text/watermark/logo" live in NEGATIVE (below), not here.
IMAGE_PROMPT_TEMPLATE = (
    "{style_prefix}, "
    "featuring {character_description}, "
    "{visual_description}, "
    "vertical 9:16 composition, cinematic lighting, ultra detailed, sharp focus"
)

# Used for scenes flagged no_character (e.g. the empty-kitchen crime shot) so the
# character is NOT injected into a "no people" environment.
IMAGE_PROMPT_NO_CHAR = (
    "{style_prefix}, "
    "{visual_description}, "
    "vertical 9:16 composition, cinematic lighting, ultra detailed, sharp focus"
)

NEGATIVE = (
    "blurry, low quality, watermark, text, logo, multiple people, "
    "distorted face, extra limbs, bad anatomy, ugly, deformed, "
    "out of frame, cropped, worst quality, low resolution, grainy"
)

# SDXL native portrait resolution (close to 9:16), must be multiples of 8
SDXL_WIDTH = 768
SDXL_HEIGHT = 1344

# Global pipe cache — loaded once, reused across all scenes
_sdxl_pipe = None


def _get_sdxl_pipe():
    global _sdxl_pipe
    if _sdxl_pipe is not None:
        return _sdxl_pipe

    try:
        import os
        import torch
        from diffusers import StableDiffusionXLPipeline

        if not torch.cuda.is_available():
            print("[Image] CUDA not available, skipping local SDXL")
            return None

        # Force HuggingFace to cache on D: drive (avoid filling C:)
        os.environ.setdefault("HF_HOME", str(SDXL_CACHE_DIR))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(SDXL_CACHE_DIR))

        print(f"[Image] Loading SDXL on {torch.cuda.get_device_name(0)}...")
        _sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir=str(SDXL_CACHE_DIR),
        )
        _sdxl_pipe = _sdxl_pipe.to("cuda")
        _sdxl_pipe.enable_vae_slicing()        # reduces VRAM for large images
        _sdxl_pipe.enable_attention_slicing()  # further VRAM saving
        print("[Image] SDXL loaded and ready on GPU")
        return _sdxl_pipe
    except Exception as e:
        print(f"[Image] SDXL load failed: {e}")
        return None


def _build_prompt(scene: dict, style: dict) -> str:
    if scene.get("no_character"):
        return IMAGE_PROMPT_NO_CHAR.format(
            style_prefix=style["prompt_prefix"],
            visual_description=scene["visual_description"],
        )
    return IMAGE_PROMPT_TEMPLATE.format(
        style_prefix=style["prompt_prefix"],
        character_description=CHARACTER_DESCRIPTION,
        visual_description=scene["visual_description"],
    )


def release_image_pipeline() -> None:
    """Free the local SD/SDXL pipelines + VRAM before the motion stage runs
    (AnimateDiff/SadTalker need the 8 GB; SD lingering caused ~50s/step)."""
    global _sdxl_pipe, _sd15_pipe
    _sdxl_pipe = None
    _sd15_pipe = None
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Image] Released local image pipeline (VRAM freed)")
    except Exception:
        pass


def _sdxl_model_ready() -> bool:
    """Return True only if all SDXL blobs are fully downloaded (no .incomplete files)."""
    blobs_dir = SDXL_CACHE_DIR / "models--stabilityai--stable-diffusion-xl-base-1.0" / "blobs"
    if not blobs_dir.exists():
        return False
    incomplete = list(blobs_dir.glob("*.incomplete"))
    return len(incomplete) == 0


def _generate_with_local_sdxl(prompt: str, output_path: Path) -> bool:
    """Local SDXL on RTX 4060 — high quality, no API needed."""
    try:
        import torch

        if not _sdxl_model_ready():
            print("[Image] SDXL still downloading — skipping (use SD 1.5 fallback)")
            return False

        pipe = _get_sdxl_pipe()
        if pipe is None:
            return False

        print("[Image] Running SDXL inference on GPU...")
        generator = torch.Generator("cuda").manual_seed(42)

        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            width=SDXL_WIDTH,
            height=SDXL_HEIGHT,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        )
        image = result.images[0]

        if image.width != VIDEO_WIDTH or image.height != VIDEO_HEIGHT:
            image = image.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.LANCZOS)

        image.save(output_path, format="PNG", optimize=False)
        return True
    except Exception as e:
        print(f"[Image] Local SDXL failed: {e}")
        return False


# ── SD 1.5 fallback (smaller model, downloads fast) ──────────────────────────
_sd15_pipe = None
SD15_WIDTH, SD15_HEIGHT = 512, 896   # 9:16 for SD 1.5


def _get_sd15_pipe():
    global _sd15_pipe
    if _sd15_pipe is not None:
        return _sd15_pipe
    try:
        import os
        import torch
        from diffusers import StableDiffusionPipeline

        if not torch.cuda.is_available():
            return None

        os.environ.setdefault("HF_HOME", str(SDXL_CACHE_DIR))
        print("[Image] Loading SD 1.5 on GPU (SDXL still downloading)...")
        _sd15_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir=str(SDXL_CACHE_DIR),
        )
        _sd15_pipe = _sd15_pipe.to("cuda")
        _sd15_pipe.enable_attention_slicing()
        print("[Image] SD 1.5 ready on GPU")
        return _sd15_pipe
    except Exception as e:
        print(f"[Image] SD 1.5 load failed: {e}")
        return None


def _generate_with_sd15(prompt: str, output_path: Path) -> bool:
    """SD 1.5 fallback — smaller model, downloads in ~5 min."""
    try:
        import torch

        pipe = _get_sd15_pipe()
        if pipe is None:
            return False

        print("[Image] Running SD 1.5 inference on GPU...")
        generator = torch.Generator("cuda").manual_seed(42)

        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            width=SD15_WIDTH,
            height=SD15_HEIGHT,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        )
        image = result.images[0]
        image = image.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.LANCZOS)
        image.save(output_path, format="PNG", optimize=False)
        return True
    except Exception as e:
        print(f"[Image] SD 1.5 failed: {e}")
        return False


def _generate_with_pollinations(prompt: str, output_path: Path) -> bool:
    """
    Pollinations.ai free tier fallback.
    Tries progressively smaller sizes; upscales result.
    """
    for width, height in [(768, 1366), (512, 912)]:
        try:
            encoded = urllib.parse.quote(prompt[:400])
            url = (
                f"https://image.pollinations.ai/prompt/{encoded}"
                f"?width={width}&height={height}&model=flux&nologo=true&nofeed=true"
            )
            resp = requests.get(url, timeout=90, stream=True)
            if resp.status_code == 402:
                print(f"[Image] Pollinations 402 at {width}x{height}, trying smaller...")
                continue
            resp.raise_for_status()
            ct = resp.headers.get("Content-Type", "")
            if "image" in ct:
                from io import BytesIO
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.LANCZOS)
                img.save(output_path, format="PNG")
                return True
        except Exception as e:
            print(f"[Image] Pollinations failed ({width}x{height}): {e}")
    return False


def _generate_placeholder(scene: dict, style: dict, output_path: Path) -> bool:
    """Styled gradient fallback — always works, no network needed."""
    try:
        import config
        style_colors = {
            "pixar_3d":    [(255, 180, 60),  (200, 80,  20)],
            "comic_book":  [(220, 40,  40),  (20,  20,  80)],
            "cinematic":   [(20,  20,  40),  (50,  25,  10)],
            "flat_motion": [(60,  160, 255), (20,  80,  180)],
        }
        style_key = next(
            (k for k, v in config.STYLE_PRESETS.items() if v["prompt_prefix"] == style["prompt_prefix"]),
            "cinematic",
        )
        c1, c2 = style_colors.get(style_key, [(30, 30, 60), (10, 10, 30)])

        img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT))
        draw = ImageDraw.Draw(img)
        for y in range(VIDEO_HEIGHT):
            t = y / VIDEO_HEIGHT
            r = int(c1[0] * (1 - t) + c2[0] * t)
            g = int(c1[1] * (1 - t) + c2[1] * t)
            b = int(c1[2] * (1 - t) + c2[2] * t)
            draw.line([(0, y), (VIDEO_WIDTH, y)], fill=(r, g, b))

        try:
            font_big = ImageFont.truetype("arial.ttf", 100)
            font_sm  = ImageFont.truetype("arial.ttf", 38)
        except Exception:
            font_big = font_sm = ImageFont.load_default()

        overlay = scene.get("text_overlay", f"SCENE {scene.get('scene_number', 1)}")
        desc = scene.get("visual_description", "")[:90]

        draw.text(
            (VIDEO_WIDTH // 2, VIDEO_HEIGHT // 2 - 60), overlay,
            font=font_big, fill=(255, 255, 255), anchor="mm",
            stroke_width=4, stroke_fill=(0, 0, 0),
        )
        draw.text(
            (VIDEO_WIDTH // 2, int(VIDEO_HEIGHT * 0.75)), desc,
            font=font_sm, fill=(210, 210, 210), anchor="mm",
            stroke_width=2, stroke_fill=(0, 0, 0),
        )

        img.save(output_path, format="PNG")
        return True
    except Exception as e:
        print(f"[Image] Placeholder failed: {e}")
        Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), (30, 30, 60)).save(output_path)
        return True


def generate_scene_image(scene: dict, style: dict, session_dir: Path) -> Path:
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out_path = images_dir / f"scene_{scene['scene_number']:02d}.png"

    if out_path.exists():
        return out_path

    prompt = _build_prompt(scene, style)
    print(f"[Image] Scene {scene['scene_number']}: generating...")

    # 0. fal.ai Flux (state-of-the-art; fixed seed keeps the character consistent)
    if PREFER_FAL_IMAGE:
        try:
            from pipeline.providers.fal_provider import fal_image, fal_available
            if fal_available() and fal_image(prompt, out_path, seed=42):
                print(f"[Image] Scene {scene['scene_number']} OK (fal Flux)")
                return out_path
        except Exception as e:
            print(f"[Image] fal Flux path error: {e}")

    # 1. Pollinations free API — Flux-quality scenes, no GPU, no key. Best FREE
    #    option and far stronger than local SD 1.5 at complex scene composition.
    if _generate_with_pollinations(prompt, out_path):
        print(f"[Image] Scene {scene['scene_number']} OK (Pollinations Flux)")
        return out_path
    time.sleep(0.5)

    # 2. Local SDXL on GPU (requires full 6.9 GB download)
    if _generate_with_local_sdxl(prompt, out_path):
        print(f"[Image] Scene {scene['scene_number']} OK (local SDXL GPU)")
        return out_path
    time.sleep(0.5)

    # 3. SD 1.5 on GPU (weakest at complex scenes — last AI fallback)
    if _generate_with_sd15(prompt, out_path):
        print(f"[Image] Scene {scene['scene_number']} OK (SD 1.5 GPU)")
        return out_path

    # 4. PIL gradient placeholder (always works, no AI)
    _generate_placeholder(scene, style, out_path)
    print(f"[Image] Scene {scene['scene_number']} OK (placeholder)")
    return out_path


def generate_all_scene_images(scenes: list[dict], style: dict, session_dir: Path) -> list[Path]:
    paths = []
    for scene in scenes:
        path = generate_scene_image(scene, style, session_dir)
        paths.append(path)
        time.sleep(0.5)
    return paths
