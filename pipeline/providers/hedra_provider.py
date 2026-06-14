"""
Hedra Character-3 talking-photo lip-sync via the public Platform API.
Key-gated and fallback-safe — returns False on any problem so the caller
falls back to local SadTalker.

Flow (api.hedra.com/web-app/public, header x-api-key):
  GET  /models                       -> pick the Character-3 model id
  POST /assets {name,type:image}     -> id ; POST /assets/{id}/upload (multipart)
  POST /assets {name,type:audio}     -> id ; POST /assets/{id}/upload (multipart)
  POST /generations {...}            -> generation id
  GET  /generations/{id}/status      -> poll until complete, download url

hedra_available() -> bool
hedra_talking_photo(image_path, audio_path, out_path, text_prompt="") -> bool
"""
import time
from pathlib import Path

import requests

from config import HEDRA_API_KEY, PREFER_HEDRA, USE_CLOUD_PROVIDERS, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS

_BASE = "https://api.hedra.com/web-app/public"
_TIMEOUT = 60
_POLL_INTERVAL = 5
_POLL_MAX = 180     # 180 * 5s = 15 min ceiling


def hedra_available() -> bool:
    return bool(USE_CLOUD_PROVIDERS and PREFER_HEDRA and HEDRA_API_KEY)


def _headers(json_body: bool = True) -> dict:
    h = {"x-api-key": HEDRA_API_KEY}
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _pick_model(session: requests.Session) -> str | None:
    r = session.get(f"{_BASE}/models", headers=_headers(False), timeout=_TIMEOUT)
    r.raise_for_status()
    models = r.json()
    if isinstance(models, dict):
        models = models.get("data") or models.get("models") or []
    if not models:
        return None
    # Prefer a Character-3 model, else first available
    for m in models:
        name = (m.get("name") or m.get("id") or "").lower()
        if "character-3" in name or "character_3" in name or "character3" in name:
            return m["id"]
    return models[0]["id"]


def _create_and_upload(session: requests.Session, path: Path, kind: str) -> str | None:
    """Create an asset of `kind` ('image'|'audio') and upload the file. Return asset id."""
    r = session.post(f"{_BASE}/assets", headers=_headers(True),
                     json={"name": Path(path).name, "type": kind}, timeout=_TIMEOUT)
    r.raise_for_status()
    asset_id = (r.json() or {}).get("id")
    if not asset_id:
        return None
    with open(path, "rb") as fh:
        up = session.post(f"{_BASE}/assets/{asset_id}/upload",
                          headers=_headers(False),
                          files={"file": (Path(path).name, fh)}, timeout=300)
    up.raise_for_status()
    return asset_id


def _fit_to_canvas(raw_mp4: Path, out_path: Path) -> bool:
    """Resize Hedra output to fill height, center-pad onto 1080x1920, NVENC encode."""
    try:
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
        print(f"[Hedra] canvas-fit failed: {e}")
        return False


def hedra_talking_photo(image_path: Path, audio_path: Path, out_path: Path,
                        text_prompt: str = "") -> bool:
    if not hedra_available():
        return False
    if not (audio_path and Path(audio_path).exists()):
        print("[Hedra] No audio for speaking scene — skipping")
        return False
    try:
        s = requests.Session()
        model_id = _pick_model(s)
        if not model_id:
            print("[Hedra] No model available")
            return False

        img_id = _create_and_upload(s, Path(image_path), "image")
        aud_id = _create_and_upload(s, Path(audio_path), "audio")
        if not (img_id and aud_id):
            print("[Hedra] Asset upload failed")
            return False

        gen = s.post(f"{_BASE}/generations", headers=_headers(True), json={
            "type": "video",
            "ai_model_id": model_id,
            "start_keyframe_id": img_id,
            "audio_id": aud_id,
            "generated_video_inputs": {
                "text_prompt": text_prompt or "",
                "resolution": "720p",
                "aspect_ratio": "9:16",
            },
        }, timeout=_TIMEOUT)
        gen.raise_for_status()
        gen_id = (gen.json() or {}).get("id")
        if not gen_id:
            print("[Hedra] No generation id")
            return False

        print(f"[Hedra] generating (id={gen_id})...")
        url = None
        for _ in range(_POLL_MAX):
            time.sleep(_POLL_INTERVAL)
            st = s.get(f"{_BASE}/generations/{gen_id}/status", headers=_headers(False), timeout=_TIMEOUT)
            st.raise_for_status()
            data = st.json() or {}
            status = (data.get("status") or "").lower()
            if status in ("complete", "completed", "success"):
                url = data.get("url") or (data.get("asset") or {}).get("url")
                break
            if status in ("error", "failed"):
                print(f"[Hedra] generation failed: {data.get('error_message') or data}")
                return False
        if not url:
            print("[Hedra] timed out waiting for generation")
            return False

        tmp = Path(out_path).with_suffix(".hedraraw.mp4")
        r = s.get(url, timeout=300, stream=True)
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)

        ok = _fit_to_canvas(tmp, out_path)
        tmp.unlink(missing_ok=True)
        if ok:
            print("[Hedra] Character-3 lip-sync OK")
        return ok
    except Exception as e:
        print(f"[Hedra] error: {e}")
        return False
