"""
One-scene SadTalker smoke test — judge talking-head lip-sync before the full render.
Requires setup_sadtalker.py to have run first.

Run: D:\round2_venv\Scripts\python.exe test_talk_scene.py
Output: outputs/motion_test/scene_speaking.mp4
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config import OUTPUTS_DIR


def main():
    print("=" * 60)
    print("  SadTalker One-Scene SPEAKING Test")
    print("=" * 60)

    out_dir = OUTPUTS_DIR / "motion_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # SadTalker needs a clean frontal face. Generate a portrait still (as production
    # does for speaking scenes) instead of reusing an old full-scene image.
    still = out_dir / "portrait_still.png"
    if not still.exists():
        print("Generating a frontal portrait still (SD 1.5)...")
        from config import STYLE_PRESETS
        from pipeline.image_generator import generate_scene_image
        from pipeline.script_generator import _PORTRAIT
        portrait_scene = {
            "scene_number": 99,
            "shot_type": "speaking",
            "visual_description": _PORTRAIT,
            "text_overlay": "TEST",
        }
        # generate into motion_test/images/scene_99.png then copy
        gen = generate_scene_image(portrait_scene, STYLE_PRESETS["cinematic"], out_dir)
        import shutil as _sh
        _sh.copy2(str(gen), str(still))

    voice = next(iter(sorted(OUTPUTS_DIR.glob("session_*/audio/voice_01.mp3"))), None)
    if not voice:
        # Generate a quick test voice line
        print("No cached voice found — generating a test voiceover...")
        from pipeline.audio_generator import generate_voiceover
        voice = OUTPUTS_DIR / "motion_test" / "test_voice.mp3"
        voice.parent.mkdir(parents=True, exist_ok=True)
        generate_voiceover(
            "His restaurant was bleeding cash, and he was the last to realize why.",
            "en-US-GuyNeural", voice,
        )

    print(f"Still : {still}")
    print(f"Voice : {voice}")

    out_path = out_dir / "scene_speaking.mp4"

    from pipeline.lipsync_generator import animate_with_sadtalker
    scene = {"scene_number": 1, "shot_type": "speaking"}
    ok = animate_with_sadtalker(Path(still), Path(voice), scene, out_path)

    print("\n" + "=" * 60)
    if ok and out_path.exists():
        print(f"  SUCCESS — review: {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)")
    else:
        print("  FAILED — see log (speaking scenes would fall back to Ken Burns).")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
