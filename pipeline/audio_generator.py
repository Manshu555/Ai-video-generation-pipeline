"""
Generates per-scene voiceover audio using edge-tts (free, no API key needed).
Optionally mixes in background music using pydub.
"""
import asyncio
import os
import time
from pathlib import Path

from config import DEFAULT_TTS_VOICE, MUSIC_DIR, MUSIC_VOLUME_DB, SCENE_SPEECH_RATE

# Narration-driven scene pacing bounds (seconds)
SCENE_MIN_DURATION = 4.0
SCENE_MAX_DURATION = 8.0
SCENE_TAIL_PADDING = 0.8   # breathing room after the voiceover ends


def get_audio_duration(path) -> float:
    """Return the duration (seconds) of an audio file, or 0.0 on failure."""
    try:
        from moviepy import AudioFileClip
        clip = AudioFileClip(str(path))
        dur = float(clip.duration)
        clip.close()
        return dur
    except Exception:
        return 0.0


def scene_duration_from_voice(voice_path) -> float:
    """
    Compute a scene's display duration from its voiceover length.
    = clamp(voice_dur + padding, MIN, MAX). Falls back to MIN if no audio.
    """
    from pathlib import Path as _P
    if voice_path and _P(str(voice_path)).exists():
        d = get_audio_duration(voice_path)
        if d > 0:
            return max(SCENE_MIN_DURATION, min(SCENE_MAX_DURATION, d + SCENE_TAIL_PADDING))
    return SCENE_MIN_DURATION


FALLBACK_VOICES = [
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-US-JennyNeural",
    "en-US-DavisNeural",
]


async def _generate_voiceover_async(text: str, voice: str, output_path: Path) -> list:
    """
    Stream TTS to mp3 AND collect per-word timing (edge-tts WordBoundary events).
    Returns a list of {"word","start","end"} in seconds.
    """
    import edge_tts
    # edge-tts 7.x defaults boundary="SentenceBoundary" (only ~2 events). We need
    # per-word events for kinetic captions, so request WordBoundary explicitly.
    communicate = edge_tts.Communicate(
        text, voice, rate=SCENE_SPEECH_RATE, boundary="WordBoundary"
    )
    timings = []
    with open(output_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                # offset/duration are in 100-nanosecond units
                start = chunk["offset"] / 1e7
                dur = chunk["duration"] / 1e7
                timings.append({"word": chunk["text"], "start": start, "end": start + dur})
    return timings


def _save_word_timings(output_path: Path, timings: list) -> None:
    import json
    jpath = output_path.with_suffix(".json")
    try:
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(timings, f, ensure_ascii=False)
    except Exception:
        pass


def load_word_timings(voice_path) -> list:
    """Load saved per-word timings for a voice mp3, or [] if missing/empty."""
    import json
    from pathlib import Path as _P
    jpath = _P(str(voice_path)).with_suffix(".json")
    if jpath.exists():
        try:
            with open(jpath, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
    return []


def _make_silent_audio(output_path: Path, duration_ms: int = 9000) -> Path:
    """Generate a silent MP3 as fallback when TTS fails."""
    try:
        from pydub import AudioSegment
        silent = AudioSegment.silent(duration=duration_ms)
        silent.export(str(output_path), format="mp3")
        return output_path
    except Exception:
        # Last resort: write minimal valid MP3 header bytes
        output_path.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 413)
        return output_path


def generate_voiceover(text: str, voice: str, output_path: Path) -> Path:
    """Generate TTS for a single voiceover line. Returns path to MP3.

    Cascade: ElevenLabs (premium, word timestamps) -> edge-tts (free, WordBoundary).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. ElevenLabs (state-of-the-art narration, returns word timings directly)
    try:
        from pipeline.providers.eleven_provider import eleven_tts, eleven_available
        if eleven_available():
            path, timings = eleven_tts(text, output_path)
            if path and output_path.exists() and output_path.stat().st_size > 1000:
                _save_word_timings(output_path, timings)
                print("[Audio] ElevenLabs narration OK")
                return output_path
    except Exception as e:
        print(f"[Audio] ElevenLabs failed, falling back to edge-tts: {e}")

    # 2. edge-tts (free) — collects WordBoundary events for kinetic captions
    voices_to_try = [voice] + [v for v in FALLBACK_VOICES if v != voice]
    for attempt_voice in voices_to_try:
        for attempt in range(2):
            try:
                timings = asyncio.run(_generate_voiceover_async(text, attempt_voice, output_path))
                if output_path.exists() and output_path.stat().st_size > 1000:
                    _save_word_timings(output_path, timings)
                    return output_path
            except Exception as e:
                print(f"[Audio] TTS failed ({attempt_voice}, attempt {attempt+1}): {e}")
                time.sleep(2)

    print(f"[Audio] All TTS attempts failed, using silent audio for: {text[:40]}...")
    return _make_silent_audio(output_path)


def generate_all_voiceovers(scenes: list[dict], voice: str, session_dir: Path) -> list[Path]:
    """Generate voiceover MP3 for every scene. Returns list of paths."""
    audio_dir = session_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for scene in scenes:
        text = scene.get("voiceover_text", "")
        out_path = audio_dir / f"voice_{scene['scene_number']:02d}.mp3"
        if not out_path.exists() or out_path.stat().st_size < 2000:
            if out_path.exists():
                out_path.unlink()  # remove corrupt/empty file
            generate_voiceover(text, voice, out_path)
            print(f"[Audio] Scene {scene['scene_number']} voiceover OK")
        paths.append(out_path)
    return paths


def mix_music_under_voice(voice_path: Path, music_path: Path, output_path: Path) -> Path:
    """
    Mix background music under a voiceover track.
    Music is looped/trimmed to match voice duration and attenuated to MUSIC_VOLUME_DB.
    """
    try:
        from pydub import AudioSegment

        voice = AudioSegment.from_file(str(voice_path))
        music = AudioSegment.from_file(str(music_path))

        # Loop music if shorter than voice
        while len(music) < len(voice):
            music = music + music
        music = music[: len(voice)]

        # Attenuate music
        music = music + MUSIC_VOLUME_DB

        # Overlay
        mixed = voice.overlay(music)
        mixed.export(str(output_path), format="mp3")
        return output_path
    except Exception as e:
        print(f"[Audio Mix] Failed, using voice only: {e}")
        return voice_path


def get_available_music_tracks() -> dict[str, Path]:
    """Return dict of {mood_label: path} for available background tracks."""
    tracks = {}
    if MUSIC_DIR.exists():
        for f in MUSIC_DIR.iterdir():
            if f.suffix.lower() in (".mp3", ".wav", ".ogg"):
                label = f.stem.replace("_", " ").title()
                tracks[label] = f
    if not tracks:
        tracks["No music"] = None
    return tracks
