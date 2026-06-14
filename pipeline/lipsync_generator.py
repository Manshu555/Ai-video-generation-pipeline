"""
Talking-head lip-sync via SadTalker, run in its OWN isolated venv as a subprocess
so its (finicky) dependencies never touch the main diffusers env.

animate_with_sadtalker(still, voice_mp3, scene, out) -> bool
  Generates a head-and-shoulders clip whose mouth/expression match the narration,
  then resizes/pads to 1080x1920 and NVENC-encodes to out_path.
  Returns False on any failure (caller falls back to Ken Burns).
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

from config import VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS, CHARACTER_REF_IMAGE

SADTALKER_DIR = Path(r"D:\sadtalker")
SADTALKER_PY = Path(r"D:\sadtalker_venv\Scripts\python.exe")
SADTALKER_SIZE = 256          # 256 (fast) or 512 (sharper, slower)
USE_ENHANCER = True           # gfpgan face enhancement

_available = None


def _sadtalker_available() -> bool:
    global _available
    if _available is not None:
        return _available
    ok = (
        SADTALKER_PY.exists()
        and (SADTALKER_DIR / "inference.py").exists()
        and (SADTALKER_DIR / "checkpoints").exists()
        and any((SADTALKER_DIR / "checkpoints").glob("*.safetensors"))
    )
    if not ok:
        print("[SadTalker] Not installed/configured — run setup_sadtalker.py "
              "(speaking scenes will fall back to Ken Burns)")
    _available = ok
    return ok


def _find_result_mp4(result_dir: Path) -> Path | None:
    mp4s = sorted(result_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return mp4s[0] if mp4s else None


def _run_sadtalker(src_img: Path, voice_path: Path, out_path: Path, scene: dict) -> bool:
    """One SadTalker attempt on a specific source image. True on success."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="sadtalker_"))
    try:
        cmd = [
            str(SADTALKER_PY), "inference.py",
            "--source_image", str(Path(src_img).resolve()),
            "--driven_audio", str(Path(voice_path).resolve()),
            "--result_dir", str(tmp_dir.resolve()),
            "--preprocess", "full",
            "--size", str(SADTALKER_SIZE),
        ]
        if USE_ENHANCER:
            cmd += ["--enhancer", "gfpgan"]

        proc = subprocess.run(
            cmd, cwd=str(SADTALKER_DIR),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=1800,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-1200:]
            print(f"[SadTalker] inference failed (rc={proc.returncode}) on {Path(src_img).name}:\n{tail}")
            return False

        raw_mp4 = _find_result_mp4(tmp_dir)
        if not raw_mp4:
            print("[SadTalker] No output mp4 produced")
            return False

        # Resize to fill height, then center-pad onto a 1080x1920 canvas, NVENC encode
        from moviepy import VideoFileClip, ColorClip, CompositeVideoClip
        from pipeline.video_generator import _get_codec

        clip = VideoFileClip(str(raw_mp4)).resized(height=VIDEO_HEIGHT)
        if clip.w > VIDEO_WIDTH:
            clip = clip.resized(width=VIDEO_WIDTH)
        bg = ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=(0, 0, 0)).with_duration(clip.duration)
        comp = CompositeVideoClip([bg, clip.with_position("center")], size=(VIDEO_WIDTH, VIDEO_HEIGHT))

        codec = _get_codec()
        extra = {"ffmpeg_params": ["-preset", "fast"]} if codec == "h264_nvenc" else {}
        comp.write_videofile(str(out_path), fps=VIDEO_FPS, codec=codec, audio=False, logger=None, **extra)
        return True
    except Exception as e:
        print(f"[SadTalker] error on {Path(src_img).name}: {e}")
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def animate_with_sadtalker(image_path: Path, voice_path: Path, scene: dict, out_path: Path) -> bool:
    if not _sadtalker_available():
        return False
    if not (voice_path and Path(voice_path).exists()):
        print("[SadTalker] No voice audio for speaking scene — skipping")
        return False

    print(f"[SadTalker] Scene {scene.get('scene_number')}: generating talking head...")
    # 1) prefer the clean reference photo (reliable face detection + consistent
    #    identity across speaking scenes); 2) fall back to the SD portrait still.
    sources = []
    if CHARACTER_REF_IMAGE and Path(CHARACTER_REF_IMAGE).exists():
        sources.append(CHARACTER_REF_IMAGE)
    sources.append(image_path)

    for src in sources:
        if _run_sadtalker(Path(src), voice_path, out_path, scene):
            return True
        print(f"[SadTalker] retrying with fallback source...")
    return False
