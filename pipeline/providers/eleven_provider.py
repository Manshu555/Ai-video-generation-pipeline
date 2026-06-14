"""
ElevenLabs narration via the `with-timestamps` REST endpoint (uses `requests`,
no extra SDK). Returns (mp3_path, word_timings) so kinetic captions stay synced.

eleven_available() -> bool
eleven_tts(text, out_mp3) -> (Path|None, list[{"word","start","end"}])
"""
import base64
from pathlib import Path

import requests

from config import (
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL,
    PREFER_ELEVENLABS, USE_CLOUD_PROVIDERS,
)

_API = "https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps"


def eleven_available() -> bool:
    return bool(USE_CLOUD_PROVIDERS and PREFER_ELEVENLABS and ELEVENLABS_API_KEY)


def _chars_to_words(chars, starts, ends) -> list:
    """Group character-level alignment into word timings on whitespace."""
    words, cur, cur_start, last_end = [], "", None, 0.0
    for ch, s, e in zip(chars, starts, ends):
        if ch.isspace():
            if cur:
                words.append({"word": cur, "start": float(cur_start), "end": float(last_end)})
                cur, cur_start = "", None
        else:
            if cur_start is None:
                cur_start = s
            cur += ch
            last_end = e
    if cur:
        words.append({"word": cur, "start": float(cur_start), "end": float(last_end)})
    return words


def eleven_tts(text: str, out_mp3: Path) -> tuple:
    """Generate narration + word timings. Returns (Path, timings) or (None, [])."""
    if not eleven_available():
        return None, []
    try:
        url = _API.format(voice=ELEVENLABS_VOICE_ID)
        headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
        body = {
            "text": text,
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": {"stability": 0.45, "similarity_boost": 0.8, "style": 0.25},
        }
        resp = requests.post(url, json=body, headers=headers, timeout=90)
        if resp.status_code != 200:
            print(f"[ElevenLabs] HTTP {resp.status_code}: {resp.text[:200]}")
            return None, []
        data = resp.json()

        audio_b64 = data.get("audio_base64")
        if not audio_b64:
            print("[ElevenLabs] No audio in response")
            return None, []
        out_mp3 = Path(out_mp3)
        out_mp3.parent.mkdir(parents=True, exist_ok=True)
        out_mp3.write_bytes(base64.b64decode(audio_b64))

        align = data.get("alignment") or data.get("normalized_alignment") or {}
        timings = _chars_to_words(
            align.get("characters", []),
            align.get("character_start_times_seconds", []),
            align.get("character_end_times_seconds", []),
        )
        print(f"[ElevenLabs] OK ({len(timings)} word timings)")
        return out_mp3, timings
    except Exception as e:
        print(f"[ElevenLabs] error: {e}")
        return None, []
