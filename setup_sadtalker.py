"""
Automated SadTalker setup (isolated from the main diffusers venv).

Does, idempotently:
  1. git clone OpenTalker/SadTalker -> D:\sadtalker
  2. create venv D:\sadtalker_venv (Python 3.11)
  3. install CUDA torch + SadTalker requirements into that venv
  4. patch the torchvision `functional_tensor` import (basicsr/facexlib) for modern torch
  5. download ~2 GB checkpoints + gfpgan weights from HF/GitHub-release mirrors

Run: D:\round2_venv\Scripts\python.exe setup_sadtalker.py
(uses only stdlib + urllib; the heavy installs happen inside the new venv)
"""
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

SADTALKER_DIR = Path(r"D:\sadtalker")
SADTALKER_VENV = Path(r"D:\sadtalker_venv")
PY311 = Path(r"C:\Users\Lenovo\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe")
PIP_CACHE = r"D:\pip_cache"
REPO_URL = "https://github.com/OpenTalker/SadTalker.git"

# Checkpoints — canonical GitHub release assets (v0.0.2-rc).
GH_BASE = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc"
CHECKPOINTS = {
    "checkpoints/mapping_00109-model.pth.tar": f"{GH_BASE}/mapping_00109-model.pth.tar",
    "checkpoints/mapping_00229-model.pth.tar": f"{GH_BASE}/mapping_00229-model.pth.tar",
    "checkpoints/SadTalker_V0.0.2_256.safetensors": f"{GH_BASE}/SadTalker_V0.0.2_256.safetensors",
    "checkpoints/SadTalker_V0.0.2_512.safetensors": f"{GH_BASE}/SadTalker_V0.0.2_512.safetensors",
}
# GFPGAN / face-detection weights (facexlib + gfpgan expect these locally)
GFPGAN_WEIGHTS = {
    "gfpgan/weights/alignment_WFLW_4HG.pth": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth",
    "gfpgan/weights/detection_Resnet50_Final.pth": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "gfpgan/weights/GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "gfpgan/weights/parsing_parsenet.pth": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
}


def run(cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kw)


def download(url: str, dest: Path, attempts: int = 4):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  [skip] {dest.name} already present")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for a in range(1, attempts + 1):
        try:
            print(f"  downloading {dest.name} (attempt {a}) ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.rename(dest)
            print(f"  done: {dest.name} ({dest.stat().st_size/1024/1024:.0f} MB)")
            return
        except Exception as e:
            last = e
            print(f"  ! failed ({e}); retrying...")
    raise RuntimeError(f"could not download {dest.name}: {last}")


def step_clone():
    if (SADTALKER_DIR / "inference.py").exists():
        print("[1/5] SadTalker repo already cloned.")
        return
    print("[1/5] Cloning SadTalker...")
    run(["git", "clone", "--depth", "1", REPO_URL, str(SADTALKER_DIR)])


def step_venv():
    if (SADTALKER_VENV / "Scripts" / "python.exe").exists():
        print("[2/5] sadtalker_venv already exists.")
        return
    print("[2/5] Creating sadtalker_venv...")
    run([str(PY311), "-m", "venv", str(SADTALKER_VENV)])


def step_install():
    py = SADTALKER_VENV / "Scripts" / "python.exe"
    env = {**os.environ, "PIP_CACHE_DIR": PIP_CACHE}
    print("[3/5] Installing CUDA torch + SadTalker deps (this is the long step)...")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"], env=env)
    # CUDA torch matching the machine (cu124)
    run([str(py), "-m", "pip", "install", "torch==2.6.0", "torchvision==0.21.0",
         "--index-url", "https://download.pytorch.org/whl/cu124"], env=env)
    # SadTalker requirements (pinned-ish, known-good with the functional_tensor patch)
    deps = [
        "numpy==1.26.4", "face_alignment==1.3.5", "imageio==2.34.0", "imageio-ffmpeg==0.4.9",
        "librosa==0.10.1", "numba", "resampy==0.4.2", "pydub==0.25.1", "scipy==1.11.4",
        "kornia==0.6.8", "yacs==0.1.8", "pyyaml", "joblib", "scikit-image", "basicsr==1.4.2",
        "facexlib==0.3.0", "gfpgan==1.3.8", "av", "safetensors", "tqdm",
    ]
    run([str(py), "-m", "pip", "install", *deps], env=env)


def step_patch():
    """Fix two known SadTalker-on-modern-stack breakages."""
    import re
    print("[4/5] Patching compatibility issues...")
    # (a) basicsr/facexlib import torchvision.transforms.functional_tensor (removed tv>=0.17)
    sp = SADTALKER_VENV / "Lib" / "site-packages"
    for rel in ["basicsr/data/degradations.py"]:
        f = sp / rel
        if f.exists():
            txt = f.read_text(encoding="utf-8")
            new = txt.replace(
                "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
                "from torchvision.transforms.functional import rgb_to_grayscale",
            )
            if new != txt:
                f.write_text(new, encoding="utf-8")
                print(f"  patched {rel}")

    # (b) deprecated numpy aliases (np.float/np.int/...) removed in numpy>=1.24
    pats = {
        r"np\.float(?![0-9a-zA-Z_])": "np.float64",
        r"np\.int(?![0-9a-zA-Z_])": "np.int64",
        r"np\.bool(?![0-9a-zA-Z_])": "bool",
        r"np\.object(?![0-9a-zA-Z_])": "object",
        r"np\.str(?![0-9a-zA-Z_])": "str",
        r"np\.complex(?![0-9a-zA-Z_])": "np.complex128",
    }
    src = SADTALKER_DIR / "src"
    n = 0
    if src.exists():
        for pf in src.rglob("*.py"):
            t = pf.read_text(encoding="utf-8", errors="ignore")
            nt = t
            for p, r in pats.items():
                nt = re.sub(p, r, nt)
            if nt != t:
                pf.write_text(nt, encoding="utf-8")
                n += 1
    print(f"  patched numpy aliases in {n} src file(s)")


def step_checkpoints():
    print("[5/5] Downloading checkpoints (~2 GB)...")
    for rel, url in CHECKPOINTS.items():
        download(url, SADTALKER_DIR / rel)
    # GFPGAN/facexlib weights are best-effort (gfpgan auto-downloads at runtime too)
    for rel, url in GFPGAN_WEIGHTS.items():
        try:
            download(url, SADTALKER_DIR / rel)
        except Exception as e:
            print(f"  [warn] optional weight {rel} skipped: {e}")


def main():
    print("=" * 60)
    print("  SadTalker Setup (isolated venv)")
    print("=" * 60)
    step_clone()
    step_venv()
    step_install()
    step_patch()
    step_checkpoints()
    print("\nSadTalker setup complete. Test with: test_talk_scene.py")


if __name__ == "__main__":
    main()
