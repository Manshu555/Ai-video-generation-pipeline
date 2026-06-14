"""
Downloads SDXL base 1.0 model weights to D:/models/sd/
Run ONCE after installing CUDA PyTorch:
    python download_sdxl.py
Model size: ~6.9 GB
"""
import os
import sys
from pathlib import Path

os.environ["HF_HOME"] = "D:/models/sd"
os.environ["TRANSFORMERS_CACHE"] = "D:/models/sd"

cache_dir = Path("D:/models/sd")
cache_dir.mkdir(parents=True, exist_ok=True)

try:
    import torch
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Install CUDA PyTorch first:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("torch not installed. Run pip install first.")
    sys.exit(1)

try:
    from diffusers import StableDiffusionXLPipeline
except ImportError:
    print("diffusers not installed. Run: pip install diffusers transformers accelerate safetensors")
    sys.exit(1)

print("Downloading SDXL base 1.0 to D:/models/sd/ ...")
print("This is ~6.9 GB and may take 10-30 minutes depending on your connection.\n")

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    cache_dir=str(cache_dir),
)

print("\nModel downloaded successfully!")
print(f"Saved to: {cache_dir}")

# Quick smoke test
print("\nRunning smoke test on GPU...")
pipe = pipe.to("cuda")
pipe.enable_vae_slicing()
pipe.enable_attention_slicing()

result = pipe(
    "cinematic portrait of a businessman, dramatic lighting, photorealistic",
    width=512,
    height=512,
    num_inference_steps=4,
    guidance_scale=1.0,
)
out = Path("D:/models/sd/smoke_test.png")
result.images[0].save(out)
print(f"Smoke test image saved: {out}")
print("SDXL is ready! Open http://localhost:8501 and run the pipeline.")
