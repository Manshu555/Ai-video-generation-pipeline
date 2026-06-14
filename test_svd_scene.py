"""
One-scene SVD smoke test — judge motion quality BEFORE the full render.

Animates an already-generated still (no image re-gen, no Ollama) with Stable
Video Diffusion and writes a single clip you can review.

Run: D:\round2_venv\Scripts\python.exe test_svd_scene.py
Output: outputs/svd_test/scene_01_motion.mp4
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HOME", "D:/models/sd")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/models/sd")

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config import OUTPUTS_DIR

# Use the cached still from the last session (already on disk)
SCENE_IMAGE = OUTPUTS_DIR / "session_20260611_143710" / "images" / "scene_01.png"
TARGET_DURATION = 5.0  # seconds for the test clip


def main():
    print("=" * 60)
    print("  SVD One-Scene Motion Test")
    print("=" * 60)

    if not SCENE_IMAGE.exists():
        # Fall back to any available scene image
        candidates = sorted(OUTPUTS_DIR.glob("session_*/images/scene_01.png"))
        if not candidates:
            print(f"ERROR: no scene image found at {SCENE_IMAGE}")
            print("Generate at least one session's images first.")
            return 1
        image_path = candidates[-1]
    else:
        image_path = SCENE_IMAGE

    print(f"Input still : {image_path}")

    out_dir = OUTPUTS_DIR / "svd_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scene_01_motion.mp4"

    from pipeline.svd_generator import animate_with_svd, MOTION_BUCKET_ID
    print(f"motion_bucket_id = {MOTION_BUCKET_ID}")
    print("First run downloads ~9.5 GB SVD weights — this can take a while.\n")

    scene = {"scene_number": 1, "camera_motion": "svd", "emotion": "tension"}
    ok = animate_with_svd(image_path, scene, out_path, TARGET_DURATION)

    print("\n" + "=" * 60)
    if ok and out_path.exists():
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  SUCCESS — review this clip:")
        print(f"  {out_path}  ({size_mb:.1f} MB, {TARGET_DURATION}s)")
        print(f"\n  If motion is too subtle, raise MOTION_BUCKET_ID in")
        print(f"  pipeline/svd_generator.py and re-run this test.")
    else:
        print("  SVD FAILED — see log above. The full pipeline would fall back")
        print("  to Ken Burns. Likely a model-download/VRAM issue.")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
