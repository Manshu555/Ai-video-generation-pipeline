"""
Direct pipeline runner — generates output.mp4 without Streamlit UI.
Run: D:\round2_venv\Scripts\python.exe generate_video.py
"""
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Must be first — sets KMP_DUPLICATE_LIB_OK and HF paths
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HOME", "D:/models/sd")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/models/sd")

# Add project to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config import STYLE_PRESETS, DEFAULT_STYLE, DEFAULT_TTS_VOICE, OUTPUTS_DIR

# ── Story choice ──────────────────────────────────────────────────────────────
STORY_ID  = 1           # 1=Dishwasher Steak (default — strong viral hook)
TONE      = "dramatic"
STYLE_KEY = "cinematic"  # cinematic | pixar_3d | comic_book | flat_motion
VOICE     = "en-US-GuyNeural"   # warm authoritative male narrator

# Set to an existing session folder name to resume (skips image generation)
# Leave None to start fresh (needed for the new 5-scene + shot_type script).
RESUME_SESSION = None

REUSE_CACHED_CLIPS = False  # False = always re-render clips with the current engines
# Motion engines: speaking scenes -> SadTalker, action scenes -> AnimateDiff,
# both fall back to Ken Burns automatically (see pipeline/video_generator.py).
# ─────────────────────────────────────────────────────────────────────────────


def load_story(story_id: int) -> dict:
    stories_file = BASE_DIR / "stories" / "templates.json"
    with open(stories_file, encoding="utf-8") as f:
        stories = json.load(f)
    return next(s for s in stories if s["id"] == story_id)


def main():
    print("\n" + "="*60)
    print("  Breakout AI — Viral Reel Generator")
    print("="*60)

    story = load_story(STORY_ID)
    style = STYLE_PRESETS[STYLE_KEY]

    print(f"\nStory  : {story['title']}")
    print(f"Style  : {style['label']}")
    print(f"Tone   : {TONE}")
    print(f"Voice  : {VOICE}")

    if RESUME_SESSION:
        session_dir = OUTPUTS_DIR / RESUME_SESSION
        print(f"\nResuming : {session_dir}")
    else:
        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = OUTPUTS_DIR / f"session_{session_ts}"
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput : {session_dir}")
    print("-"*60)

    # ── Step 1: Script ────────────────────────────────────────────────────────
    script_path = session_dir / "script.json"
    if script_path.exists() and RESUME_SESSION:
        print("\n[1/5] Script (cached)...")
        from pipeline.script_generator import load_script
        scenes = load_script(session_dir)
    else:
        print("\n[1/5] Generating script with Ollama...")
        from pipeline.script_generator import generate_script
        scenes = generate_script(story, TONE, style, session_dir)
    print(f"      {len(scenes)} scenes")

    # ── Step 2: Images ────────────────────────────────────────────────────────
    images_dir = session_dir / "images"
    existing_images = sorted(images_dir.glob("scene_*.png")) if images_dir.exists() else []
    if len(existing_images) >= len(scenes) and RESUME_SESSION:
        print("\n[2/5] Images (cached)...")
        image_paths = existing_images[:len(scenes)]
    else:
        print("\n[2/5] Generating scene images (SDXL GPU -> SD 1.5 -> Pollinations)...")
        from pipeline.image_generator import generate_all_scene_images
        image_paths = generate_all_scene_images(scenes, style, session_dir)
    print(f"      {len(image_paths)} images ready")

    # ── Step 3: Voiceovers (BEFORE clips — narration drives scene length) ──────
    print("\n[3/5] Generating voiceovers (edge-tts)...")
    from pipeline.audio_generator import (
        generate_all_voiceovers, scene_duration_from_voice, get_audio_duration,
    )
    voice_paths = generate_all_voiceovers(scenes, VOICE, session_dir)
    # Speaking scenes: clip length = voice length (SadTalker matches audio).
    # Action scenes: voice length + padding (clamped).
    target_durations = []
    for sc, vp in zip(scenes, voice_paths):
        if sc.get("shot_type") == "speaking":
            d = get_audio_duration(vp) or scene_duration_from_voice(vp)
        else:
            d = scene_duration_from_voice(vp)
        target_durations.append(round(d, 2))
    print(f"      {len(voice_paths)} voiceovers ready")
    print(f"      Scene durations (voice-driven): "
          f"{', '.join(f'{d:.1f}s' for d in target_durations)}")

    # Free the local SD/SDXL pipeline + VRAM before the motion stage (AnimateDiff /
    # SadTalker need the full 8 GB; a resident SD pipe caused ~50s/step slowdown).
    from pipeline.image_generator import release_image_pipeline
    release_image_pipeline()

    # ── Step 4: Video clips ─────────────────────────────────────────────────────
    # speaking -> Hedra (cloud) -> SadTalker (local); action -> Kling (cloud) ->
    # AnimateDiff (local); both fall back to Ken Burns. Routed by shot_type.
    clips_dir = session_dir / "clips"
    existing_clips = sorted(clips_dir.glob("scene_*.mp4")) if clips_dir.exists() else []
    if len(existing_clips) >= len(scenes) and RESUME_SESSION and REUSE_CACHED_CLIPS:
        print("\n[4/5] Clips (cached)...")
        clip_paths = existing_clips[:len(scenes)]
    else:
        print("\n[4/5] Animating scenes (Hedra/Kling cloud -> local fallback)...")
        from pipeline.video_generator import animate_all_scenes
        clip_paths = animate_all_scenes(
            image_paths, scenes, session_dir,
            target_durations=target_durations, style=style, voice_paths=voice_paths,
        )
    print(f"      {len(clip_paths)} clips ready")

    # ── Step 5: Assemble ──────────────────────────────────────────────────────
    print("\n[5/5] Assembling final reel...")
    from pipeline.assembler import assemble_reel

    # Background music (optional — use first available track)
    from config import MUSIC_DIR
    music_path = None
    if MUSIC_DIR.exists():
        tracks = list(MUSIC_DIR.glob("*.mp3"))
        if tracks:
            music_path = tracks[0]
            print(f"      Music: {music_path.name}")

    # Optional: generate a cinematic bed via Stable Audio if enabled + no local track
    if music_path is None:
        from config import PREFER_FAL_MUSIC
        if PREFER_FAL_MUSIC:
            try:
                from pipeline.providers.fal_provider import fal_music, fal_available
                if fal_available():
                    bed = session_dir / "audio" / "music_bed.mp3"
                    if fal_music("cinematic emotional minor-key piano underscore that resolves "
                                 "into warm hopeful strings, sparse, filmic", bed,
                                 sum(target_durations)):
                        music_path = bed
                        print("      Music: generated cinematic bed (Stable Audio)")
            except Exception as e:
                print(f"      Music gen skipped: {e}")

    final_path = assemble_reel(
        clip_paths=clip_paths,
        voice_paths=voice_paths,
        scenes=scenes,
        session_dir=session_dir,
        music_path=music_path,
        output_filename="output.mp4",
    )

    print("\n" + "="*60)
    print(f"  DONE! Video saved to:")
    print(f"  {final_path}")
    print("="*60 + "\n")

    # Copy to project root for easy access
    import shutil
    root_output = BASE_DIR / "output.mp4"
    shutil.copy2(str(final_path), str(root_output))
    print(f"  Also copied to: {root_output}")

    return final_path


if __name__ == "__main__":
    main()
