"""
Pexels stock-footage provider (free API key). Powers the faceless B-roll flow:
search a vertical clip matching a scene's stock query, cover-crop it to 1080x1920,
fit it to the narration duration, and NVENC-encode.

Key-gated + fallback-safe: returns False on missing key / no match / any error, so
the caller falls back to Ken Burns on the still.

pexels_available() -> bool
fetch_stock_clip(query, out_path, target_duration) -> bool
"""
import tempfile
from pathlib import Path

import requests

from config import PEXELS_API_KEY, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS

_SEARCH = "https://api.pexels.com/videos/search"
_TIMEOUT = 30


def pexels_available() -> bool:
    # Free stock API — independent of the paid-cloud master switch.
    return bool(PEXELS_API_KEY)


def _pick_video_file(videos: list) -> str | None:
    """From Pexels search results pick the best PORTRAIT mp4 link (closest to 1080x1920)."""
    best, best_score = None, -1.0
    target_ar = VIDEO_HEIGHT / VIDEO_WIDTH  # 1.777...
    for v in videos:
        for f in v.get("video_files", []):
            w, h = f.get("width") or 0, f.get("height") or 0
            link = f.get("link")
            ftype = (f.get("file_type") or "")
            if not link or "mp4" not in ftype or h <= w:
                continue  # need a portrait mp4
            # prefer height near 1920, aspect near 9:16, not absurdly large
            ar = h / max(1, w)
            ar_pen = abs(ar - target_ar)
            res_pen = abs(h - VIDEO_HEIGHT) / VIDEO_HEIGHT
            score = 2.0 - ar_pen - 0.5 * res_pen + (0.3 if h >= 1280 else 0)
            if score > best_score:
                best, best_score = link, score
    return best


def fetch_stock_clip(query: str, out_path: Path, target_duration: float) -> bool:
    if not pexels_available():
        return False
    query = (query or "").strip()
    if not query:
        return False
    tmp = None
    try:
        headers = {"Authorization": PEXELS_API_KEY}
        params = {"query": query, "orientation": "portrait", "size": "medium", "per_page": 15}
        r = requests.get(_SEARCH, headers=headers, params=params, timeout=_TIMEOUT)
        if r.status_code != 200:
            print(f"[Pexels] HTTP {r.status_code} for '{query}': {r.text[:160]}")
            return False
        videos = (r.json() or {}).get("videos", [])
        link = _pick_video_file(videos)
        if not link:
            print(f"[Pexels] No portrait clip for '{query}'")
            return False

        tmp = Path(tempfile.mkstemp(suffix=".mp4", prefix="pexels_")[1])
        with requests.get(link, timeout=180, stream=True) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    f.write(chunk)

        return _fit_clip(tmp, out_path, target_duration)
    except Exception as e:
        print(f"[Pexels] error for '{query}': {e}")
        return False
    finally:
        if tmp:
            try:
                Path(tmp).unlink(missing_ok=True)
            except Exception:
                pass   # Windows may briefly hold the handle; OS cleans temp later


def _fit_clip(src_mp4: Path, out_path: Path, target_duration: float) -> bool:
    """Loop/trim to target_duration, cover-crop to 1080x1920, drop audio, NVENC encode."""
    base = final = None
    try:
        from moviepy import VideoFileClip, concatenate_videoclips
        from pipeline.video_generator import _get_codec

        base = VideoFileClip(str(src_mp4)).without_audio()

        # Loop or trim to the narration length
        if base.duration < target_duration:
            loops = int(target_duration / base.duration) + 1
            final = concatenate_videoclips([base] * loops).subclipped(0, target_duration)
        else:
            final = base.subclipped(0, target_duration)

        # Cover-fit to 1080x1920 (scale to fill, center-crop)
        scale = max(VIDEO_WIDTH / final.w, VIDEO_HEIGHT / final.h)
        final = final.resized(scale)
        final = final.cropped(
            x_center=final.w / 2, y_center=final.h / 2,
            width=VIDEO_WIDTH, height=VIDEO_HEIGHT,
        )

        codec = _get_codec()
        extra = {"ffmpeg_params": ["-preset", "fast"]} if codec == "h264_nvenc" else {}
        final.write_videofile(str(out_path), fps=VIDEO_FPS, codec=codec, audio=False, logger=None, **extra)
        return True
    except Exception as e:
        print(f"[Pexels] fit/encode failed: {e}")
        return False
    finally:
        # Close readers so the temp source file can be deleted on Windows
        for c in (final, base):
            try:
                if c is not None:
                    c.close()
            except Exception:
                pass
