"""
One-scene ModelScope text-to-video smoke test — judge quality before enabling
USE_MODELSCOPE for the full render.
Run: D:\round2_venv\Scripts\python.exe test_modelscope_scene.py
Output: outputs/motion_test/scene_modelscope.mp4
First run downloads ~3.5 GB to D:/models/sd.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HOME", "D:/models/sd")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/models/sd")

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config import OUTPUTS_DIR, STYLE_PRESETS

TARGET_DURATION = 6.0


def main():
    print("=" * 60)
    print("  ModelScope One-Scene text-to-video Test")
    print("=" * 60)

    style = STYLE_PRESETS["cinematic"]
    # Mirrors the S2 "kitchen" action beat (no_character so no person is forced in)
    scene = {
        "scene_number": 1,
        "shot_type": "action",
        "no_character": True,
        "visual_description": ("interior of a commercial restaurant kitchen at night, stainless steel "
                               "counters, raw premium steaks on a tray, an open back door to a dark alley, "
                               "steam rising, cinematic wide shot"),
        "motion_prompt": "slow cinematic pan across the kitchen toward the dark exit",
        "emotion": "shock",
    }

    out_dir = OUTPUTS_DIR / "motion_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scene_modelscope.mp4"

    from pipeline.modelscope_generator import animate_with_modelscope, MS_FRAMES, MS_SIZE
    print(f"frames={MS_FRAMES}, size={MS_SIZE}x{MS_SIZE}, target={TARGET_DURATION}s")
    print("First run downloads ~3.5 GB.\n")

    ok = animate_with_modelscope(scene, style, out_path, TARGET_DURATION)

    print("\n" + "=" * 60)
    if ok and out_path.exists():
        print(f"  SUCCESS — review: {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)")
    else:
        print("  FAILED — see log above (would fall back to Ken Burns).")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
