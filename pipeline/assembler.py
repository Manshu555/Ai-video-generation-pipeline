"""
Assembles scene clips + audio into a final 9:16 vertical reel using MoviePy v2.
Features:
- Semi-transparent caption bars (not raw text on video)
- Cinematic color grade pass
- Fade transitions between scenes
- GPU encoding (NVENC) when available
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

from config import VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS


def _render_text_frame(
    text: str,
    width: int,
    font_size: int = 52,
    bar_opacity: int = 160,  # 0-255
    padding: int = 24,
) -> PILImage.Image:
    """
    Render text on a transparent image with a semi-transparent dark bar behind it.
    Returns RGBA PIL image of (width, auto_height).
    """
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    # Wrap text to fit width
    dummy = PILImage.new("RGBA", (1, 1))
    d = ImageDraw.Draw(dummy)
    max_chars = max(10, int(width * 0.85 / (font_size * 0.55)))
    import textwrap
    lines = textwrap.wrap(text, width=max_chars) or [text]

    line_h = font_size + 6
    total_h = line_h * len(lines) + padding * 2

    img = PILImage.new("RGBA", (width, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Dark bar background
    draw.rectangle([(0, 0), (width, total_h)], fill=(0, 0, 0, bar_opacity))

    y = padding
    for line in lines:
        draw.text(
            (width // 2, y),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            anchor="mt",
            stroke_width=2,
            stroke_fill=(0, 0, 0, 200),
        )
        y += line_h

    return img


def _render_title_frame(text: str, width: int) -> PILImage.Image:
    """Bold ALL-CAPS title card at top of frame."""
    try:
        font = ImageFont.truetype("arialbd.ttf", 72)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 72)
        except Exception:
            font = ImageFont.load_default()

    padding = 20
    import textwrap
    lines = textwrap.wrap(text.upper(), width=max(8, int(width * 0.75 / 40))) or [text.upper()]
    line_h = 78
    total_h = line_h * len(lines) + padding * 2

    img = PILImage.new("RGBA", (width, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (width, total_h)], fill=(0, 0, 0, 140))

    y = padding
    for line in lines:
        draw.text(
            (width // 2, y),
            line,
            font=font,
            fill=(255, 230, 60, 255),  # gold-yellow title
            anchor="mt",
            stroke_width=3,
            stroke_fill=(0, 0, 0, 220),
        )
        y += line_h

    return img


def _composite_text_onto_frame(video_frame: np.ndarray, text_img: PILImage.Image, y_pos: int) -> np.ndarray:
    """Paste an RGBA text image onto a video frame (HxWx3 numpy) at given y."""
    h, w = video_frame.shape[:2]
    tw, th = text_img.size

    # Clip to frame bounds
    if y_pos + th > h:
        y_pos = h - th
    if y_pos < 0:
        y_pos = 0

    frame = PILImage.fromarray(video_frame).convert("RGBA")
    frame.paste(text_img, (0, y_pos), text_img)
    return np.array(frame.convert("RGB"))


# ── Kinetic captions (word-by-word, synced to speech) ─────────────────────────
def _render_caption_stage(words: list[str], highlight_idx: int, width: int,
                          font_size: int = 58) -> PILImage.Image:
    """Render the caption showing words[0..highlight_idx], newest word in gold."""
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    import textwrap
    shown = words[: highlight_idx + 1]
    text = " ".join(shown)
    max_chars = max(10, int(width * 0.86 / (font_size * 0.55)))
    lines = textwrap.wrap(text, width=max_chars) or [text]

    line_h = font_size + 10
    pad = 28
    total_h = line_h * len(lines) + pad * 2
    img = PILImage.new("RGBA", (width, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([(28, 0), (width - 28, total_h)], radius=24, fill=(0, 0, 0, 150))

    last_word = shown[-1] if shown else ""
    y = pad
    for line in lines:
        # center each line; color the final word gold when it's on this line
        total_w = draw.textlength(line, font=font)
        x = (width - total_w) / 2
        parts = line.split(" ")
        for wi, wtok in enumerate(parts):
            is_last = (line is lines[-1]) and (wi == len(parts) - 1) and (wtok == last_word)
            color = (255, 215, 60, 255) if is_last else (255, 255, 255, 255)
            draw.text((x, y), wtok, font=font, fill=color,
                      stroke_width=2, stroke_fill=(0, 0, 0, 220))
            x += draw.textlength(wtok + " ", font=font)
        y += line_h
    return img


def _build_caption_stages(text: str, timings: list, duration: float, width: int):
    """Return (start_times[], stage_imgs[]) for progressive word reveal."""
    words = (text or "").split()
    if not words:
        return [], []
    if timings:
        starts = [float(t.get("start", 0.0)) for t in timings][: len(words)]
        if len(starts) < len(words):  # pad evenly if fewer boundaries than words
            step = duration / max(1, len(words))
            starts += [starts[-1] + step * (i + 1) for i in range(len(words) - len(starts))]
    else:
        step = duration / max(1, len(words))
        starts = [i * step for i in range(len(words))]
    stages = [_render_caption_stage(words, i, width) for i in range(len(words))]
    return starts, stages


# ── Cinematic FX ──────────────────────────────────────────────────────────────
_LIGHT_LEAK = None


def _light_leak(width: int, height: int) -> np.ndarray:
    """Precomputed warm radial glow (HxWx3 float) added subtly per frame."""
    global _LIGHT_LEAK
    if _LIGHT_LEAK is not None:
        return _LIGHT_LEAK
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width * 0.78, height * 0.22  # upper-right glow
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    g = np.clip(1.0 - d / (width * 0.9), 0, 1) ** 2
    leak = np.zeros((height, width, 3), dtype="float32")
    leak[:, :, 0] = g * 60   # warm R
    leak[:, :, 1] = g * 32   # G
    leak[:, :, 2] = g * 10   # B
    _LIGHT_LEAK = leak
    return _LIGHT_LEAK


def _apply_cinematic_fx(frame: np.ndarray, t: float, grain: float = 7.0) -> np.ndarray:
    """Film grain + pulsing warm light-leak. frame HxWx3 uint8 -> uint8."""
    h, w = frame.shape[:2]
    out = frame.astype("float32")
    # pulsing light leak
    pulse = 0.6 + 0.4 * (0.5 + 0.5 * np.sin(t * 1.3))
    out += _light_leak(w, h) * pulse
    # film grain
    if grain > 0:
        noise = np.random.normal(0, grain, (h, w, 1)).astype("float32")
        out += noise
    return np.clip(out, 0, 255).astype("uint8")


def _zoom_punch(frame: np.ndarray, t: float, dur: float = 0.3, amt: float = 0.06) -> np.ndarray:
    """Quick scale-in over the first `dur` seconds of a scene."""
    if t >= dur:
        return frame
    p = t / dur
    scale = 1.0 + amt * (1.0 - (p * p * (3 - 2 * p)))  # eased 1+amt -> 1.0
    h, w = frame.shape[:2]
    nw, nh = int(w * scale), int(h * scale)
    img = PILImage.fromarray(frame).resize((nw, nh), PILImage.BILINEAR)
    x, y = (nw - w) // 2, (nh - h) // 2
    return np.array(img.crop((x, y, x + w, y + h)))


# ── Hero text (scene 4 climax — full-frame kinetic typography) ────────────────
def _font(size: int, bold: bool = True):
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf", "arialbd.ttf"]
    for nm in names:
        try:
            return ImageFont.truetype(nm, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _scale_alpha(layer: PILImage.Image, a: float) -> PILImage.Image:
    """Return the RGBA layer with its alpha channel multiplied by a in [0,1]."""
    if a >= 1.0:
        return layer
    r, g, b, al = layer.split()
    al = al.point(lambda v: int(v * max(0.0, min(1.0, a))))
    return PILImage.merge("RGBA", (r, g, b, al))


def _apply_hero_text(frame: np.ndarray, t: float, line1: str, line2: str) -> np.ndarray:
    """
    Full-frame climax treatment: darken the frame, fade in line1, then reveal
    line2 word-by-word with a scale-punch. line2 in gold. (Scene 4 only.)
    """
    h, w = frame.shape[:2]
    dark_a = min(1.0, t / 0.3) * 0.6                      # black overlay fades in
    base = np.clip(frame.astype("float32") * (1.0 - dark_a), 0, 255).astype("uint8")
    img = PILImage.fromarray(base).convert("RGBA")

    f1 = _font(54)
    f2 = _font(112)

    # line 1 — fades in 0.3 .. 1.0s
    a1 = max(0.0, min(1.0, (t - 0.3) / 0.7))
    if a1 > 0 and line1:
        layer = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.text((w // 2, int(h * 0.40)), line1.upper(), font=f1, anchor="mm",
               fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))
        img = PILImage.alpha_composite(img, _scale_alpha(layer, a1))

    # line 2 — words appear at 1.2s, 2.2s, ... each with a brief scale-punch
    words2 = (line2 or "").upper().split()
    reveal = [1.2 + i * 1.0 for i in range(len(words2))]
    shown = [wd for wd, rt in zip(words2, reveal) if t >= rt]
    if shown:
        last_rt = max(rt for rt in reveal if t >= rt)
        since = t - last_rt
        punch = 1.0 + 0.18 * max(0.0, 1.0 - since / 0.25)   # eased 1.18 -> 1.0
        layer = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.text((w // 2, int(h * 0.55)), " ".join(shown), font=f2, anchor="mm",
               fill=(255, 215, 60, 255), stroke_width=4, stroke_fill=(0, 0, 0, 255))
        if punch > 1.001:
            nw, nh = int(w * punch), int(h * punch)
            layer = layer.resize((nw, nh), PILImage.BILINEAR).crop(
                ((nw - w) // 2, (nh - h) // 2, (nw - w) // 2 + w, (nh - h) // 2 + h))
        img = PILImage.alpha_composite(img, layer)

    return np.array(img.convert("RGB"))


def _fit_audio(src, dur: float):
    """Loop/trim an AudioFileClip to exactly `dur` seconds."""
    from moviepy import concatenate_audioclips
    if src.duration >= dur:
        return src.subclipped(0, dur)
    loops = int(dur / src.duration) + 1
    return concatenate_audioclips([src] * loops).subclipped(0, dur)


def _check_nvenc() -> bool:
    try:
        import subprocess
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        return "h264_nvenc" in r.stdout
    except Exception:
        return False


_NVENC_AVAILABLE = None


def _get_codec() -> str:
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is None:
        _NVENC_AVAILABLE = _check_nvenc()
    return "h264_nvenc" if _NVENC_AVAILABLE else "libx264"


def assemble_reel(
    clip_paths: list[Path],
    voice_paths: list[Path],
    scenes: list[dict],
    session_dir: Path,
    music_path: Path | None = None,
    output_filename: str = "final_reel.mp4",
) -> Path:
    from moviepy import (
        VideoFileClip,
        AudioFileClip,
        CompositeVideoClip,
        concatenate_videoclips,
        CompositeAudioClip,
        VideoClip,
    )
    from moviepy import vfx

    final_dir = session_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    output_path = final_dir / output_filename

    # Pre-render gold title cards per scene
    title_imgs = {}
    for scene in scenes:
        sn = scene["scene_number"]
        overlay_text = scene.get("text_overlay", "")
        if overlay_text:
            title_imgs[sn] = _render_title_frame(overlay_text, VIDEO_WIDTH)

    from pipeline.audio_generator import (
        scene_duration_from_voice, get_audio_duration, load_word_timings,
    )

    from pipeline.video_generator import _apply_color_grade  # centralized color arc

    CROSSFADE = 0.15  # short crossfade between scenes (no fade-to-black gap)
    assembled_clips = []
    seg_durs = []     # per-scene durations (for the per-scene music arc)
    seg_attens = []   # per-scene music attenuation in dB
    n = len(scenes)

    for idx, (clip_path, voice_path, scene) in enumerate(zip(clip_paths, voice_paths, scenes)):
        vclip = VideoFileClip(str(clip_path))
        vclip = vclip.resized((VIDEO_WIDTH, VIDEO_HEIGHT))

        # Narration-driven duration — speaking scenes = voice length (talking head
        # already matches audio); action scenes = voice + padding.
        if scene.get("shot_type") == "speaking":
            target_dur = get_audio_duration(voice_path) or scene_duration_from_voice(voice_path)
        else:
            target_dur = scene_duration_from_voice(voice_path)
        sn = scene["scene_number"]
        seg_durs.append(target_dur)
        seg_attens.append(float(scene.get("music_attenuation", -22.0)))

        # Duration: match video to target (loop if clip shorter)
        if vclip.duration < target_dur:
            from moviepy import concatenate_videoclips as _cv
            loops = int(target_dur / vclip.duration) + 1
            vclip = _cv([vclip] * loops).subclipped(0, target_dur)
        else:
            vclip = vclip.subclipped(0, target_dur)

        emotion = scene.get("emotion", "neutral")
        hero = bool(scene.get("hero_text"))
        hero_l1 = scene.get("hero_line1", "")
        hero_l2 = scene.get("hero_line2", "")

        if hero:
            # ── Scene 4 climax: full-frame hero typography, no title/caption bar ──
            orig_make_frame = vclip.get_frame

            def _make_overlay_frame(t, _omf=orig_make_frame, _emo=emotion,
                                    _l1=hero_l1, _l2=hero_l2):
                frame = _omf(t)
                frame = _apply_color_grade(frame, _emo)
                frame = _apply_cinematic_fx(frame, t, grain=4.0)
                return _apply_hero_text(frame, t, _l1, _l2)
        else:
            # ── Standard scene: gold title + word-by-word kinetic captions ──
            t_img = title_imgs.get(sn)
            timings = load_word_timings(voice_path)
            starts, stages = _build_caption_stages(
                scene.get("voiceover_text", ""), timings, target_dur, VIDEO_WIDTH
            )
            title_y = int(VIDEO_HEIGHT * 0.05)
            caption_y = int(VIDEO_HEIGHT * 0.78)
            orig_make_frame = vclip.get_frame

            def _make_overlay_frame(t, _omf=orig_make_frame, _ti=t_img, _emo=emotion,
                                    _starts=starts, _stages=stages, _ty=title_y, _cy=caption_y):
                frame = _omf(t)
                frame = _zoom_punch(frame, t)                 # quick scale-in at scene start
                frame = _apply_color_grade(frame, _emo)       # cold→warm emotional arc
                frame = _apply_cinematic_fx(frame, t)         # grain + warm light leak
                if _ti is not None:
                    frame = _composite_text_onto_frame(frame, _ti, _ty)
                # current kinetic caption stage = last word whose start <= t
                if _stages:
                    k = -1
                    for i, st in enumerate(_starts):
                        if t >= st:
                            k = i
                        else:
                            break
                    if k >= 0:
                        frame = _composite_text_onto_frame(frame, _stages[k], _cy)
                return frame

        vclip = VideoClip(_make_overlay_frame, duration=target_dur)
        vclip = vclip.with_fps(VIDEO_FPS)

        # Add voiceover (kept full-length — target_dur already fits voice + padding)
        if voice_path and voice_path.exists():
            audio = AudioFileClip(str(voice_path))
            if audio.duration > target_dur:
                audio = audio.subclipped(0, target_dur)
            vclip = vclip.with_audio(audio)

        # Transitions: gentle fade-in on the very first scene, crossfade into
        # every subsequent scene; gentle fade-out on the last scene.
        effects = []
        if idx == 0:
            effects.append(vfx.FadeIn(0.3))
        else:
            effects.append(vfx.CrossFadeIn(CROSSFADE))
        if idx == n - 1:
            effects.append(vfx.FadeOut(0.3))
        vclip = vclip.with_effects(effects)
        assembled_clips.append(vclip)

    # Concatenate with negative padding so crossfades overlap (no black gaps)
    final = concatenate_videoclips(assembled_clips, method="compose", padding=-CROSSFADE)

    # Background music — per-scene attenuation arc (tension bed under S1-S3,
    # near-silence under the S4 climax, warm resolution on S5). Falls back to a
    # flat bed, then to no music, on any error.
    if music_path and music_path.exists():
        try:
            from moviepy import concatenate_audioclips
            segs = []
            for dur_i, atten_db in zip(seg_durs, seg_attens):
                src = AudioFileClip(str(music_path))
                vol = 10 ** (atten_db / 20.0)        # dB -> linear gain
                segs.append(_fit_audio(src, dur_i).multiply_volume(vol))
            bed = concatenate_audioclips(segs)
            if bed.duration > final.duration:
                bed = bed.subclipped(0, final.duration)
            if final.audio is not None:
                final = final.with_audio(CompositeAudioClip([final.audio, bed]))
            print("[Assembler] Music arc mixed (per-scene attenuation)")
        except Exception as e:
            print(f"[Assembler] Music arc failed ({e}); trying flat bed...")
            try:
                from moviepy import concatenate_audioclips as _ca
                music = AudioFileClip(str(music_path))
                loops = int(final.duration / music.duration) + 1
                music_looped = _ca([music] * loops).subclipped(0, final.duration)
                if final.audio is not None:
                    final = final.with_audio(CompositeAudioClip(
                        [final.audio, music_looped.multiply_volume(0.07)]))
            except Exception as e2:
                print(f"[Assembler] Music mix failed, skipping: {e2}")

    codec = _get_codec()
    extra_params = ["-preset", "fast"] if codec == "h264_nvenc" else ["-preset", "slow", "-crf", "18"]

    print(f"[Assembler] Encoding with {codec}...")
    final.write_videofile(
        str(output_path),
        fps=VIDEO_FPS,
        codec=codec,
        audio_codec="aac",
        audio_bitrate="192k",
        temp_audiofile=str(session_dir / "temp_audio.m4a"),
        remove_temp=True,
        ffmpeg_params=extra_params,
        logger="bar",
    )

    print(f"[Assembler] Final reel saved: {output_path}")
    return output_path
