"""
Quick smoke test — verifies all pipeline stages work end-to-end.
Run: python test_pipeline.py
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

PASS = "[PASS]"
FAIL = "[FAIL]"


def test_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"{PASS} CUDA: {name} ({vram:.1f} GB VRAM)")
            return True
        else:
            print(f"{FAIL} CUDA not available — PyTorch version: {torch.__version__}")
            print("       Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
            return False
    except ImportError:
        print(f"{FAIL} torch not installed")
        return False


def test_sdxl_available():
    model_dir = Path("D:/models/sd")
    snapshots = list(model_dir.rglob("model_index.json"))
    if snapshots:
        print(f"{PASS} SDXL weights found: {snapshots[0].parent}")
        return True
    else:
        print(f"{FAIL} SDXL not downloaded yet — run: python download_sdxl.py")
        return False


def test_ollama():
    try:
        import requests
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2:3b",
                "messages": [{"role": "user", "content": "Say OK in one word."}],
                "stream": False,
                "options": {"num_predict": 5},
            },
            timeout=30,
        )
        resp.raise_for_status()
        msg = resp.json()["message"]["content"]
        print(f"{PASS} Ollama llama3.2:3b: '{msg.strip()}'")
        return True
    except Exception as e:
        print(f"{FAIL} Ollama: {e}")
        return False


def test_nvenc():
    import subprocess
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        if "h264_nvenc" in r.stdout:
            print(f"{PASS} NVENC GPU encoding available")
            return True
        else:
            print(f"[WARN] NVENC not in FFmpeg — will use libx264 CPU encoding")
            return False
    except Exception as e:
        print(f"[WARN] FFmpeg check failed: {e}")
        return False


def test_image_gen():
    from pathlib import Path
    out = Path("D:/pip_tmp/test_scene.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pipeline.image_generator import generate_scene_image
    from config import STYLE_PRESETS

    scene = {
        "scene_number": 1,
        "visual_description": "businessman looking at camera, dramatic studio lighting",
        "text_overlay": "TEST SCENE",
        "voiceover_text": "This is a test.",
        "camera_motion": "slow push-in",
        "emotion": "dramatic",
    }
    style = STYLE_PRESETS["cinematic"]

    # Use a temp session dir
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_scene_image(scene, style, Path(tmpdir))
        size_kb = path.stat().st_size / 1024
        if size_kb > 50:
            print(f"{PASS} Image gen: {size_kb:.0f} KB at {path.name}")
            return True
        else:
            print(f"{FAIL} Image gen: only {size_kb:.0f} KB (likely placeholder)")
            return False


def test_ken_burns():
    from pathlib import Path
    import tempfile
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # Create a simple test image
    from PIL import Image as PILImage
    import numpy as np
    img = PILImage.fromarray(np.random.randint(50, 200, (1920, 1080, 3), dtype=np.uint8))

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        img.save(img_path)

        from pipeline.video_generator import _animate_ken_burns
        out = Path(tmpdir) / "test_clip.mp4"
        scene = {"scene_number": 1, "duration_seconds": 3, "camera_motion": "slow push-in", "emotion": "dramatic"}
        ok = _animate_ken_burns(img_path, scene, out)
        if ok and out.exists():
            size_kb = out.stat().st_size / 1024
            print(f"{PASS} Ken Burns: {size_kb:.0f} KB, 3s clip generated")
            return True
        else:
            print(f"{FAIL} Ken Burns failed")
            return False


def test_edge_tts():
    import asyncio
    import tempfile
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test_voice.mp3"
        from pipeline.audio_generator import generate_voiceover
        generate_voiceover("This is a professional test of the voice system.", "en-US-GuyNeural", out)
        size_kb = out.stat().st_size / 1024
        if size_kb > 5:
            print(f"{PASS} edge-tts: {size_kb:.0f} KB MP3")
            return True
        else:
            print(f"{FAIL} edge-tts: tiny output ({size_kb:.0f} KB)")
            return False


if __name__ == "__main__":
    print("=" * 55)
    print("  Pipeline Smoke Test")
    print("=" * 55)

    results = {}
    results["cuda"]     = test_cuda()
    results["sdxl"]     = test_sdxl_available()
    results["nvenc"]    = test_nvenc()
    results["ollama"]   = test_ollama()
    results["imggen"]   = test_image_gen()
    results["kburn"]    = test_ken_burns()
    results["tts"]      = test_edge_tts()

    print("=" * 55)
    passed = sum(results.values())
    total  = len(results)
    print(f"  Results: {passed}/{total} passed")

    if not results["cuda"]:
        print("\nNEXT STEP: PyTorch CUDA is still installing.")
        print("  Wait for the pip install to complete, then run this test again.")
    elif not results["sdxl"]:
        print("\nNEXT STEP: Run  python download_sdxl.py  to download SDXL (6.9 GB).")
    else:
        print("\nAll systems ready! Open http://localhost:8501")
