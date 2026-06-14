"""
Local action-motion via ModelScope text-to-video (damo-vilab/text-to-video-ms-1.7b).

Like AnimateDiff this is TEXT-to-video: it generates frames from the scene prompt
and does NOT animate the curated still. Output is 256x256 and carries a faint
training-data watermark, so it is OPT-IN (config.USE_MODELSCOPE, default off) and
sits behind Kling / above AnimateDiff in the action cascade. Ken Burns on a good
still is usually sharper.

Tuned for 8 GB VRAM:
  - enable_model_cpu_offload()   (only the active module on GPU)
  - enable_vae_slicing()         (cut VAE decode memory)

On any failure animate_with_modelscope() returns False and the caller falls back
to Ken Burns. First call downloads ~3.5 GB to SDXL_CACHE_DIR (anonymous).
"""
import os
from pathlib import Path

from config import VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS, SDXL_CACHE_DIR, MODELSCOPE_REPO

# ── Tunables ──────────────────────────────────────────────────────────────────
MS_FRAMES = 16      # frames generated (drop to 8 if OOM)
MS_STEPS = 25       # inference steps
MS_SIZE = 256       # native resolution of this model (square)
SEED = 42
INTERP_FACTOR = 2   # 2x linear blend interpolation for smoother motion

_ms_pipe = None
_ms_failed = False


def _get_modelscope_pipe():
    global _ms_pipe, _ms_failed
    if _ms_pipe is not None:
        return _ms_pipe
    if _ms_failed:
        return None
    try:
        import torch
        from diffusers import DiffusionPipeline

        if not torch.cuda.is_available():
            print("[ModelScope] CUDA not available")
            _ms_failed = True
            return None

        os.environ.setdefault("HF_HOME", str(SDXL_CACHE_DIR))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(SDXL_CACHE_DIR))

        print(f"[ModelScope] Loading {MODELSCOPE_REPO} (first run downloads ~3.5 GB)...")
        try:
            pipe = DiffusionPipeline.from_pretrained(
                MODELSCOPE_REPO, torch_dtype=torch.float16, variant="fp16",
                cache_dir=str(SDXL_CACHE_DIR),
            )
        except Exception:
            # fp16 variant not available — fall back to default weights in fp16
            pipe = DiffusionPipeline.from_pretrained(
                MODELSCOPE_REPO, torch_dtype=torch.float16, cache_dir=str(SDXL_CACHE_DIR),
            )
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        pipe.enable_model_cpu_offload()
        _ms_pipe = pipe
        print("[ModelScope] Ready")
        return _ms_pipe
    except Exception as e:
        print(f"[ModelScope] Load failed: {e}")
        _ms_failed = True
        return None


def _frames_to_uint8_list(frames):
    """Normalize diffusers T2V output to a list of HxWx3 uint8 numpy frames.

    Handles both shapes: a batched np.ndarray (b, f, h, w, c) in [0,1] and a
    list-of-list of PIL images (frames[0] = the first video's frame list).
    """
    import numpy as np
    seq = frames[0]            # unwrap batch -> first video's frames
    out = []
    for fr in seq:
        if isinstance(fr, np.ndarray):
            arr = fr
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255).astype("uint8") if arr.max() <= 1.0 \
                    else np.clip(arr, 0, 255).astype("uint8")
            out.append(arr)
        else:                  # PIL.Image
            out.append(np.array(fr.convert("RGB")))
    return out


def _cover(arr):
    """Cover-fit a frame to 1080x1920 (scale to fill, center-crop) — avoids the
    extreme vertical stretch a direct resize of a 256x256 square would cause."""
    import numpy as np
    from PIL import Image as PILImage
    img = PILImage.fromarray(arr).convert("RGB")
    w, h = img.size
    scale = max(VIDEO_WIDTH / w, VIDEO_HEIGHT / h)
    nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
    img = img.resize((nw, nh), PILImage.LANCZOS)
    x, y = (nw - VIDEO_WIDTH) // 2, (nh - VIDEO_HEIGHT) // 2
    img = img.crop((x, y, x + VIDEO_WIDTH, y + VIDEO_HEIGHT))
    return np.array(img)


def animate_with_modelscope(scene: dict, style: dict, out_path: Path, target_duration: float) -> bool:
    """ModelScope text-to-video action motion -> mp4 of length target_duration. False on failure."""
    try:
        import torch
        from moviepy import ImageSequenceClip

        from pipeline.image_generator import _build_prompt, NEGATIVE
        from pipeline.animatediff_generator import _interpolate  # reuse blend-interp

        pipe = _get_modelscope_pipe()
        if pipe is None:
            return False

        prompt = _build_prompt(scene, style)
        print(f"[ModelScope] Scene {scene.get('scene_number')}: generating {MS_FRAMES} frames...")
        generator = torch.Generator("cuda").manual_seed(SEED)
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            num_frames=MS_FRAMES,
            num_inference_steps=MS_STEPS,
            height=MS_SIZE,
            width=MS_SIZE,
            generator=generator,
        )

        frames = _frames_to_uint8_list(result.frames)
        up = [_cover(f) for f in frames]
        # Boomerang for a seamless loop, then blend-interpolate for smoothness
        boomerang = up + up[-2:0:-1] if len(up) > 2 else up
        smooth = _interpolate(boomerang, INTERP_FACTOR)

        seq_fps = max(1.0, len(smooth) / float(target_duration))
        clip = ImageSequenceClip(smooth, fps=seq_fps).with_duration(target_duration)

        from pipeline.video_generator import _get_codec
        codec = _get_codec()
        extra = {"ffmpeg_params": ["-preset", "fast"]} if codec == "h264_nvenc" else {}
        clip.write_videofile(str(out_path), fps=VIDEO_FPS, codec=codec, audio=False, logger=None, **extra)
        return True
    except Exception as e:
        print(f"[ModelScope] Scene {scene.get('scene_number')} failed: {e}")
        return False
    finally:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
