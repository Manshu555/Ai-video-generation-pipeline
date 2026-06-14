"""
fal.ai clients: Flux (images), Kling (image-to-video), Stable Audio (music bed).
Key-gated and fallback-safe — every function returns False/None on any problem.

fal_available() -> bool
fal_image(prompt, out_path, seed=42)           -> bool   (saves 1080x1920 PNG)
fal_image_to_video(image_path, motion_prompt, out_path, seconds) -> bool
fal_music(prompt, out_path, seconds)           -> bool
"""
import os
from pathlib import Path

import requests

from config import (
    FAL_KEY, FAL_FLUX_MODEL, FAL_KLING_MODEL, FAL_MUSIC_MODEL,
    USE_CLOUD_PROVIDERS, VIDEO_WIDTH, VIDEO_HEIGHT,
)

# Flux ~1MP, 9:16-ish portrait
_FLUX_W, _FLUX_H = 768, 1344


def _fal():
    """Import fal_client lazily; set the key in env (the SDK reads FAL_KEY)."""
    try:
        import fal_client
    except Exception as e:
        print(f"[fal] fal-client not installed: {e}")
        return None
    if FAL_KEY:
        os.environ["FAL_KEY"] = FAL_KEY
    return fal_client


def fal_available() -> bool:
    return bool(USE_CLOUD_PROVIDERS and FAL_KEY)


def _download(url: str, out_path: Path) -> bool:
    try:
        r = requests.get(url, timeout=180, stream=True)
        r.raise_for_status()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        return out_path.stat().st_size > 1000
    except Exception as e:
        print(f"[fal] download failed: {e}")
        return False


def fal_image(prompt: str, out_path: Path, seed: int = 42) -> bool:
    """Generate a still via Flux; save a 1080x1920 PNG. Fixed seed = consistency."""
    if not fal_available():
        return False
    fc = _fal()
    if fc is None:
        return False
    try:
        result = fc.subscribe(FAL_FLUX_MODEL, arguments={
            "prompt": prompt,
            "image_size": {"width": _FLUX_W, "height": _FLUX_H},
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "num_images": 1,
            "enable_safety_checker": True,
            "seed": seed,
        })
        imgs = (result or {}).get("images") or []
        if not imgs:
            print("[fal] Flux returned no images")
            return False
        tmp = Path(out_path).with_suffix(".falraw")
        if not _download(imgs[0]["url"], tmp):
            return False
        from PIL import Image
        img = Image.open(tmp).convert("RGB").resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.LANCZOS)
        img.save(out_path, format="PNG")
        tmp.unlink(missing_ok=True)
        print("[fal] Flux image OK")
        return True
    except Exception as e:
        print(f"[fal] Flux error: {e}")
        return False


def fal_image_to_video(image_path: Path, motion_prompt: str, out_path: Path,
                       seconds: float = 5.0) -> bool:
    """Animate a still with Kling (cinematic motion + subject consistency)."""
    if not fal_available():
        return False
    fc = _fal()
    if fc is None:
        return False
    try:
        image_url = fc.upload_file(str(Path(image_path).resolve()))
        duration = "10" if seconds > 7 else "5"
        result = fc.subscribe(FAL_KLING_MODEL, arguments={
            "image_url": image_url,
            "prompt": motion_prompt or "subtle natural cinematic motion, slow camera move",
            "duration": duration,
            "aspect_ratio": "9:16",
            "negative_prompt": "blur, distort, warp, morphing, extra limbs, flicker",
            "cfg_scale": 0.5,
        })
        video = (result or {}).get("video") or {}
        url = video.get("url")
        if not url:
            print("[fal] Kling returned no video url")
            return False
        ok = _download(url, out_path)
        if ok:
            print("[fal] Kling image-to-video OK")
        return ok
    except Exception as e:
        print(f"[fal] Kling error: {e}")
        return False


def fal_music(prompt: str, out_path: Path, seconds: float = 30.0) -> bool:
    """Generate a cinematic instrumental bed via Stable Audio."""
    if not fal_available():
        return False
    fc = _fal()
    if fc is None:
        return False
    try:
        result = fc.subscribe(FAL_MUSIC_MODEL, arguments={
            "prompt": prompt,
            "seconds_total": int(max(5, min(47, seconds))),
        })
        result = result or {}
        node = result.get("audio_file") or result.get("audio") or {}
        url = node.get("url") if isinstance(node, dict) else None
        if not url:
            print("[fal] Stable Audio returned no url")
            return False
        ok = _download(url, out_path)
        if ok:
            print("[fal] Stable Audio music OK")
        return ok
    except Exception as e:
        print(f"[fal] Stable Audio error: {e}")
        return False
