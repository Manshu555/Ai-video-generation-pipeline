"""
Animates scene stills into video clips using cinematic Ken Burns effect.
Uses GPU-accelerated encoding (NVENC) when available, falls back to libx264.
"""
import os
import time
from pathlib import Path

from config import (
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_FPS,
    COLOR_GRADES,
)


def _ease_in_out(t: float) -> float:
    """Smooth cubic ease-in-out: starts/ends slow, fast in the middle."""
    return t * t * (3.0 - 2.0 * t)


def _check_nvenc() -> bool:
    """Return True if FFmpeg on PATH has h264_nvenc support."""
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


_NVENC_AVAILABLE = None


def _get_codec() -> str:
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is None:
        _NVENC_AVAILABLE = _check_nvenc()
        print(f"[Video] Codec: {'h264_nvenc (GPU)' if _NVENC_AVAILABLE else 'libx264 (CPU)'}")
    return "h264_nvenc" if _NVENC_AVAILABLE else "libx264"


def _apply_vignette(arr, strength: float = 0.45):
    """Add a dark circular vignette to the frame (numpy array HxWx3)."""
    import numpy as np
    h, w = arr.shape[:2]
    y_lin = (np.linspace(-1, 1, h) ** 2).reshape(h, 1)
    x_lin = (np.linspace(-1, 1, w) ** 2).reshape(1, w)
    mask = 1.0 - strength * np.clip(x_lin + y_lin, 0, 1)
    return np.clip(arr * mask[:, :, np.newaxis], 0, 255).astype("uint8")


def _apply_color_grade(arr, emotion: str = "neutral"):
    """
    Per-emotion color grade — backed by config.COLOR_GRADES (the cold→warm arc).
    Returns numpy array same shape/dtype. Centralized in the assembler so every
    motion engine (Hedra/Kling/AnimateDiff/Ken Burns) gets one consistent grade.
    """
    import numpy as np
    r_mult, g_mult, b_mult = COLOR_GRADES.get(emotion, (1.0, 1.0, 1.0))
    out = arr.astype("float32")
    out[:, :, 0] = np.clip(out[:, :, 0] * r_mult, 0, 255)
    out[:, :, 1] = np.clip(out[:, :, 1] * g_mult, 0, 255)
    out[:, :, 2] = np.clip(out[:, :, 2] * b_mult, 0, 255)
    return out.astype("uint8")


def _animate_ken_burns(image_path: Path, scene: dict, out_path: Path) -> bool:
    """
    Cinematic Ken Burns with:
    - Smooth cubic ease-in-out motion
    - Per-emotion color grading
    - Vignette overlay
    - GPU (NVENC) or CPU encoding
    """
    try:
        import numpy as np
        from moviepy import VideoClip
        from PIL import Image as PILImage

        duration = float(scene.get("duration_seconds", 9))
        motion = scene.get("camera_motion", "slow push-in")
        emotion = scene.get("emotion", "neutral")

        # Load and prepare base image
        pil_img = (
            PILImage.open(str(image_path))
            .convert("RGB")
            .resize((VIDEO_WIDTH, VIDEO_HEIGHT), PILImage.LANCZOS)
        )
        base_arr = np.array(pil_img)
        # Color grade is applied once in the assembler (per scene emotion) so it's
        # consistent across all motion engines — not here, to avoid double-grading.

        def make_frame(t: float):
            progress = _ease_in_out(t / duration)   # smooth easing
            img = PILImage.fromarray(base_arr)

            if motion in ("slow push-in", "zoom out"):
                scale = (1.0 + 0.10 * progress) if motion == "slow push-in" else (1.10 - 0.10 * progress)
                new_w = int(VIDEO_WIDTH * scale)
                new_h = int(VIDEO_HEIGHT * scale)
                img = img.resize((new_w, new_h), PILImage.BILINEAR)
                x = (new_w - VIDEO_WIDTH) // 2
                y = (new_h - VIDEO_HEIGHT) // 2
                img = img.crop((x, y, x + VIDEO_WIDTH, y + VIDEO_HEIGHT))

            elif motion in ("pan left", "pan right"):
                padding = int(VIDEO_WIDTH * 0.08)
                wide_w = VIDEO_WIDTH + padding
                img = img.resize((wide_w, VIDEO_HEIGHT), PILImage.BILINEAR)
                x = int(padding * progress) if motion == "pan left" else int(padding * (1 - progress))
                img = img.crop((x, 0, x + VIDEO_WIDTH, VIDEO_HEIGHT))

            elif motion in ("tilt up", "tilt down"):
                padding = int(VIDEO_HEIGHT * 0.08)
                tall_h = VIDEO_HEIGHT + padding
                img = img.resize((VIDEO_WIDTH, tall_h), PILImage.BILINEAR)
                y = int(padding * (1 - progress)) if motion == "tilt up" else int(padding * progress)
                img = img.crop((0, y, VIDEO_WIDTH, y + VIDEO_HEIGHT))

            frame = np.array(img.resize((VIDEO_WIDTH, VIDEO_HEIGHT)))
            return _apply_vignette(frame)

        clip = VideoClip(make_frame, duration=duration)

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
        return True
    except Exception as e:
        print(f"[Ken Burns] Failed: {e}")
        return False


def animate_scene(
    image_path: Path,
    scene: dict,
    session_dir: Path,
    target_duration: float | None = None,
    style: dict | None = None,
    voice_path: Path | None = None,
) -> Path:
    clips_dir = session_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    out_path = clips_dir / f"scene_{scene['scene_number']:02d}.mp4"
    if out_path.exists():
        return out_path

    dur = float(target_duration) if target_duration else float(scene.get("duration_seconds", 6))
    shot = scene.get("shot_type", "action")
    print(f"[Video Gen] Scene {scene['scene_number']} ({shot}, target {dur:.1f}s)...")

    sn = scene["scene_number"]

    # 1a. Speaking scene -> Hedra Character-3 (cloud) -> SadTalker (local)
    if shot == "speaking":
        from config import PREFER_HEDRA
        if PREFER_HEDRA:
            try:
                from pipeline.providers.hedra_provider import hedra_talking_photo, hedra_available
                if hedra_available() and hedra_talking_photo(image_path, voice_path, out_path):
                    print(f"[Video Gen] Scene {sn} OK (Hedra Character-3)")
                    return out_path
            except Exception as e:
                print(f"[Video Gen] Hedra path error: {e}")
        try:
            from pipeline.lipsync_generator import animate_with_sadtalker
            if animate_with_sadtalker(image_path, voice_path, scene, out_path):
                print(f"[Video Gen] Scene {sn} OK (SadTalker lip-sync)")
                return out_path
        except Exception as e:
            print(f"[Video Gen] SadTalker path error: {e}")

    # 1b. Action scene -> Kling image-to-video (cloud) -> AnimateDiff (local)
    else:
        from config import PREFER_FAL_VIDEO
        if PREFER_FAL_VIDEO:
            try:
                from pipeline.providers.fal_provider import fal_image_to_video, fal_available
                if fal_available() and fal_image_to_video(
                    image_path, scene.get("motion_prompt", ""), out_path, dur
                ):
                    print(f"[Video Gen] Scene {sn} OK (fal Kling image-to-video)")
                    return out_path
            except Exception as e:
                print(f"[Video Gen] fal Kling path error: {e}")
        from config import USE_MODELSCOPE
        if USE_MODELSCOPE:
            try:
                from pipeline.modelscope_generator import animate_with_modelscope
                if animate_with_modelscope(scene, style or {}, out_path, dur):
                    print(f"[Video Gen] Scene {sn} OK (ModelScope text-to-video)")
                    return out_path
            except Exception as e:
                print(f"[Video Gen] ModelScope path error: {e}")

        from config import USE_ANIMATEDIFF
        if USE_ANIMATEDIFF:
            try:
                from pipeline.animatediff_generator import animate_with_animatediff
                if animate_with_animatediff(scene, style or {}, out_path, dur):
                    print(f"[Video Gen] Scene {sn} OK (AnimateDiff motion)")
                    return out_path
            except Exception as e:
                print(f"[Video Gen] AnimateDiff path error: {e}")

    # 2. Cinematic Ken Burns — automatic fallback (never hard-fails)
    scene_for_kb = dict(scene)
    scene_for_kb["duration_seconds"] = dur
    if _animate_ken_burns(image_path, scene_for_kb, out_path):
        print(f"[Video Gen] Scene {scene['scene_number']} OK (Ken Burns fallback)")
        return out_path

    # 3. Last-resort: static clip
    try:
        from moviepy import ImageClip
        clip = ImageClip(str(image_path), duration=dur)
        clip.write_videofile(str(out_path), fps=VIDEO_FPS, codec="libx264", audio=False, logger=None)
        print(f"[Video Gen] Scene {scene['scene_number']} OK (static fallback)")
    except Exception as e:
        print(f"[Video Gen] Static fallback also failed: {e}")

    return out_path


def animate_all_scenes(
    image_paths: list[Path],
    scenes: list[dict],
    session_dir: Path,
    target_durations: list[float] | None = None,
    style: dict | None = None,
    voice_paths: list[Path] | None = None,
) -> list[Path]:
    clip_paths = []
    for i, (image_path, scene) in enumerate(zip(image_paths, scenes)):
        td = target_durations[i] if target_durations else None
        vp = voice_paths[i] if voice_paths else None
        path = animate_scene(image_path, scene, session_dir,
                             target_duration=td, style=style, voice_path=vp)
        clip_paths.append(path)
        time.sleep(0.5)
    return clip_paths
