"""
Local image-to-video using Stable Video Diffusion (SVD) on the RTX 4060.
Animates a generated still into a short clip with REAL motion (not just zoom/pan).

Tuned for 8 GB VRAM:
  - enable_model_cpu_offload()  (only active module on GPU)
  - vae.enable_slicing()        (cut VAE decode memory)
  - decode_chunk_size=2         (decode frames in small chunks)

SVD-XT is gated on HF; with no HF_TOKEN we pull from a non-gated community
mirror. If nothing loads, animate_with_svd() returns False and the caller
falls back to Ken Burns.
"""
import os
from pathlib import Path

from config import VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS, SDXL_CACHE_DIR

# ── Tunables ──────────────────────────────────────────────────────────────────
# SVD-safe portrait gen size (9:16). Drop to (512, 896) if you hit OOM.
SVD_WIDTH, SVD_HEIGHT = 576, 1024
NUM_FRAMES = 25
DECODE_CHUNK_SIZE = 2
# Motion amount knob — higher = more movement. 127 default; 160-200 = bigger action.
MOTION_BUCKET_ID = 140
NOISE_AUG_STRENGTH = 0.02
SEED = 42

# Candidate repos tried in order. Gated official repo only works with HF_TOKEN.
_MIRROR_REPOS = [
    "vdo/stable-video-diffusion-img2vid-xt-1-1",   # non-gated community mirror
    "stabilityai/stable-video-diffusion-img2vid-xt",  # official (needs HF_TOKEN)
    "stabilityai/stable-video-diffusion-img2vid",     # official 14-frame (needs token)
]

_svd_pipe = None
_svd_failed = False  # once we know it can't load, stop retrying every scene


def _get_svd_pipe():
    global _svd_pipe, _svd_failed
    if _svd_pipe is not None:
        return _svd_pipe
    if _svd_failed:
        return None

    try:
        import torch
        from diffusers import StableVideoDiffusionPipeline

        if not torch.cuda.is_available():
            print("[SVD] CUDA not available — cannot run SVD")
            _svd_failed = True
            return None

        os.environ.setdefault("HF_HOME", str(SDXL_CACHE_DIR))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(SDXL_CACHE_DIR))
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        last_err = None
        for repo in _MIRROR_REPOS:
            # Skip gated official repos when we have no token
            if repo.startswith("stabilityai/") and not hf_token:
                print(f"[SVD] Skipping gated repo {repo} (no HF_TOKEN)")
                continue
            try:
                print(f"[SVD] Loading {repo} on {torch.cuda.get_device_name(0)} ...")
                kwargs = dict(
                    torch_dtype=torch.float16,
                    cache_dir=str(SDXL_CACHE_DIR),
                )
                if hf_token:
                    kwargs["token"] = hf_token
                # Try fp16 variant first, then default weights
                try:
                    pipe = StableVideoDiffusionPipeline.from_pretrained(
                        repo, variant="fp16", **kwargs
                    )
                except Exception:
                    pipe = StableVideoDiffusionPipeline.from_pretrained(repo, **kwargs)

                pipe.enable_model_cpu_offload()
                try:
                    pipe.vae.enable_slicing()
                except Exception:
                    pass
                _svd_pipe = pipe
                print(f"[SVD] Ready ({repo})")
                return _svd_pipe
            except Exception as e:
                last_err = e
                print(f"[SVD] Failed to load {repo}: {e}")

        print(f"[SVD] No SVD repo could be loaded. Last error: {last_err}")
        _svd_failed = True
        return None
    except Exception as e:
        print(f"[SVD] Pipe init failed: {e}")
        _svd_failed = True
        return None


def _prepare_image(image_path: Path):
    """Load still and resize to SVD gen size."""
    from PIL import Image as PILImage
    return (
        PILImage.open(str(image_path))
        .convert("RGB")
        .resize((SVD_WIDTH, SVD_HEIGHT), PILImage.LANCZOS)
    )


def animate_with_svd(image_path: Path, scene: dict, out_path: Path, target_duration: float) -> bool:
    """
    Animate a still with SVD and write an mp4 of length `target_duration`.
    Returns True on success, False on any failure (caller falls back to Ken Burns).
    """
    try:
        import numpy as np
        import torch
        from PIL import Image as PILImage
        from moviepy import ImageSequenceClip

        pipe = _get_svd_pipe()
        if pipe is None:
            return False

        image = _prepare_image(image_path)
        print(f"[SVD] Generating motion for scene {scene.get('scene_number')} "
              f"(motion_bucket_id={MOTION_BUCKET_ID}, {NUM_FRAMES} frames)...")

        generator = torch.Generator("cuda").manual_seed(SEED)
        result = pipe(
            image,
            decode_chunk_size=DECODE_CHUNK_SIZE,
            num_frames=NUM_FRAMES,
            motion_bucket_id=MOTION_BUCKET_ID,
            noise_aug_strength=NOISE_AUG_STRENGTH,
            generator=generator,
        )
        frames = result.frames[0]  # list of PIL images

        # Upscale frames to full 1080x1920 and convert to numpy
        up = [
            np.array(f.convert("RGB").resize((VIDEO_WIDTH, VIDEO_HEIGHT), PILImage.LANCZOS))
            for f in frames
        ]

        # Boomerang: forward + reverse (drop duplicate end frames) for seamless, longer motion
        boomerang = up + up[-2:0:-1]

        # Retime so the whole sequence lasts exactly target_duration
        seq_fps = max(1.0, len(boomerang) / float(target_duration))
        clip = ImageSequenceClip(boomerang, fps=seq_fps).with_duration(target_duration)

        # Encode (NVENC when available)
        from pipeline.video_generator import _get_codec
        codec = _get_codec()
        extra = {"ffmpeg_params": ["-preset", "fast"]} if codec == "h264_nvenc" else {}
        clip.write_videofile(
            str(out_path),
            fps=VIDEO_FPS,
            codec=codec,
            audio=False,
            logger=None,
            **extra,
        )

        # Free VRAM between scenes
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        return True
    except Exception as e:
        print(f"[SVD] animate failed for scene {scene.get('scene_number')}: {e}")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return False
