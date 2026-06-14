"""
Local action-motion via AnimateDiff (SD 1.5 + motion adapter) on the RTX 4060.
Generates a short clip with REAL body/camera motion from the scene prompt.

Tuned for 8 GB VRAM:
  - enable_model_cpu_offload()   (only active module on GPU)
  - enable_vae_slicing()         (cut VAE decode memory)

No HF token needed — the motion adapter (~1.7 GB) downloads anonymously and
reuses the already-cached SD 1.5. On any failure animate_with_animatediff()
returns False and the caller falls back to Ken Burns.
"""
import os
from pathlib import Path

from config import VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS, SDXL_CACHE_DIR

# ── Tunables ──────────────────────────────────────────────────────────────────
SD15_REPO = "runwayml/stable-diffusion-v1-5"
MOTION_ADAPTER_REPO = "guoyww/animatediff-motion-adapter-v1-5-3"
# Local dir with directly-downloaded adapter (avoids HF xet stalls). Preferred if present.
MOTION_ADAPTER_LOCAL = Path("D:/models/animatediff_adapter")
GEN_WIDTH, GEN_HEIGHT = 512, 768   # portrait-ish; drop to 512x512 if OOM
NUM_FRAMES = 24
STEPS = 24
GUIDANCE = 7.5
SEED = 42
INTERP_FACTOR = 2   # 2x linear blend interpolation for smoother motion

_ad_pipe = None
_ad_failed = False


def _get_animatediff_pipe():
    global _ad_pipe, _ad_failed
    if _ad_pipe is not None:
        return _ad_pipe
    if _ad_failed:
        return None
    try:
        import torch
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

        if not torch.cuda.is_available():
            print("[AnimateDiff] CUDA not available")
            _ad_failed = True
            return None

        os.environ.setdefault("HF_HOME", str(SDXL_CACHE_DIR))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(SDXL_CACHE_DIR))

        adapter_src = (
            str(MOTION_ADAPTER_LOCAL)
            if (MOTION_ADAPTER_LOCAL / "diffusion_pytorch_model.safetensors").exists()
            else MOTION_ADAPTER_REPO
        )
        print(f"[AnimateDiff] Loading motion adapter from {adapter_src} ...")
        adapter = MotionAdapter.from_pretrained(
            adapter_src, torch_dtype=torch.float16, cache_dir=str(SDXL_CACHE_DIR)
        )
        print(f"[AnimateDiff] Loading SD 1.5 + adapter on {torch.cuda.get_device_name(0)}...")
        pipe = AnimateDiffPipeline.from_pretrained(
            SD15_REPO, motion_adapter=adapter,
            torch_dtype=torch.float16, cache_dir=str(SDXL_CACHE_DIR),
        )
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear", clip_sample=False, timestep_spacing="linspace",
        )
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        pipe.enable_model_cpu_offload()
        _ad_pipe = pipe
        print("[AnimateDiff] Ready")
        return _ad_pipe
    except Exception as e:
        print(f"[AnimateDiff] Load failed: {e}")
        _ad_failed = True
        return None


def _interpolate(frames, factor: int):
    """Cheap linear blend interpolation between consecutive numpy frames."""
    import numpy as np
    if factor <= 1 or len(frames) < 2:
        return frames
    out = []
    for i in range(len(frames) - 1):
        a = frames[i].astype("float32")
        b = frames[i + 1].astype("float32")
        for k in range(factor):
            t = k / factor
            out.append((a * (1 - t) + b * t).astype("uint8"))
    out.append(frames[-1])
    return out


def animate_with_animatediff(scene: dict, style: dict, out_path: Path, target_duration: float) -> bool:
    """AnimateDiff action motion -> mp4 of length target_duration. False on failure."""
    try:
        import numpy as np
        import torch
        from PIL import Image as PILImage
        from moviepy import ImageSequenceClip

        from pipeline.image_generator import _build_prompt, NEGATIVE

        pipe = _get_animatediff_pipe()
        if pipe is None:
            return False

        prompt = _build_prompt(scene, style)
        print(f"[AnimateDiff] Scene {scene.get('scene_number')}: generating {NUM_FRAMES} frames...")
        generator = torch.Generator("cuda").manual_seed(SEED)
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            num_frames=NUM_FRAMES,
            height=GEN_HEIGHT,
            width=GEN_WIDTH,
            guidance_scale=GUIDANCE,
            num_inference_steps=STEPS,
            generator=generator,
        )
        frames = result.frames[0]  # list of PIL images

        # Upscale to full 9:16 and to numpy
        up = [
            np.array(f.convert("RGB").resize((VIDEO_WIDTH, VIDEO_HEIGHT), PILImage.LANCZOS))
            for f in frames
        ]
        # Boomerang for seamless loop, then blend-interpolate for smoothness
        boomerang = up + up[-2:0:-1]
        smooth = _interpolate(boomerang, INTERP_FACTOR)

        seq_fps = max(1.0, len(smooth) / float(target_duration))
        clip = ImageSequenceClip(smooth, fps=seq_fps).with_duration(target_duration)

        from pipeline.video_generator import _get_codec
        codec = _get_codec()
        extra = {"ffmpeg_params": ["-preset", "fast"]} if codec == "h264_nvenc" else {}
        clip.write_videofile(str(out_path), fps=VIDEO_FPS, codec=codec, audio=False, logger=None, **extra)

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"[AnimateDiff] Scene {scene.get('scene_number')} failed: {e}")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return False
