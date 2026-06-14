"""
One-scene AnimateDiff smoke test — judge ACTION motion before the full render.
Run: D:\round2_venv\Scripts\python.exe test_motion_scene.py
Output: outputs/motion_test/scene_action.mp4
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
    print("  AnimateDiff One-Scene ACTION Motion Test")
    print("=" * 60)

    style = STYLE_PRESETS["cinematic"]
    scene = {
        "scene_number": 1,
        "shot_type": "action",
        "visual_description": "a man in a chef's apron rushing through a chaotic restaurant kitchen at night, grabbing plates, steam rising, dramatic motion",
        "emotion": "tension",
    }

    out_dir = OUTPUTS_DIR / "motion_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scene_action.mp4"

    from pipeline.animatediff_generator import animate_with_animatediff, NUM_FRAMES
    print(f"num_frames={NUM_FRAMES}, target={TARGET_DURATION}s")
    print("First run downloads the ~1.7 GB motion adapter.\n")

    ok = animate_with_animatediff(scene, style, out_path, TARGET_DURATION)

    print("\n" + "=" * 60)
    if ok and out_path.exists():
        print(f"  SUCCESS — review: {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)")
    else:
        print("  FAILED — see log above (would fall back to Ken Burns).")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
