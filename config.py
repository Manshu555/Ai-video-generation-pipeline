import os
from pathlib import Path

# Required on Windows to avoid OMP duplicate library crash with numpy+torch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Cache HuggingFace models on D: drive (avoid filling C:)
os.environ.setdefault("HF_HOME", "D:/models/sd")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/models/sd")
# Disable HF's Rust "xet" downloader — it stalls/panics on this machine
# ("memory allocation ... failed"); the standard HTTP downloader is reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── API Keys (optional — pipeline falls back to local/free engines if absent) ─
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
FAL_KEY = os.getenv("FAL_KEY", "")                      # fal.ai: Flux images, Kling i2v, Stable Audio
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")  # premium narration + word timestamps
HEDRA_API_KEY = os.getenv("HEDRA_API_KEY", "")            # Character-3 talking-photo lip-sync
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
GOOGLE_DRIVE_CREDENTIALS_FILE = os.getenv("GOOGLE_DRIVE_CREDENTIALS_FILE", "credentials.json")

# ── Cloud-provider switches ───────────────────────────────────────────────────
# Master switch + per-stage preferences. A stage with no key auto-disables and
# falls back to the local engine, so an empty .env runs fully local with no crash.
# LOCAL MODE: fal balance exhausted + no ElevenLabs/Hedra keys, so run fully local
# (SD 1.5 images, AnimateDiff/Ken Burns motion, SadTalker lip-sync, edge-tts voice).
# Flip USE_CLOUD_PROVIDERS back to True after adding credits/keys to .env.
USE_CLOUD_PROVIDERS = False
PREFER_FAL_IMAGE = True     # Flux/Ideogram via fal.ai for stills (else local SDXL/SD1.5)
PREFER_FAL_VIDEO = True     # Kling image-to-video for action scenes (else AnimateDiff/Ken Burns)
PREFER_HEDRA = True         # Hedra Character-3 lip-sync for speaking scenes (else SadTalker)
PREFER_ELEVENLABS = True    # ElevenLabs narration (else edge-tts)
PREFER_FAL_MUSIC = False    # Stable Audio music bed via fal.ai (else bundled CC track)
# AnimateDiff is text-to-video (ignores the curated still) and low-coherence on
# 8 GB VRAM — it produced grain textures. Off by default: action scenes use
# cinematic Ken Burns on the good still instead. Enable only with more VRAM.
USE_ANIMATEDIFF = False
# ModelScope text-to-video (damo-vilab/text-to-video-ms-1.7b) — also text-to-video,
# 256x256 + watermarked, so softer than Ken Burns on a good still. Opt-in; sits
# above AnimateDiff in the action cascade. First use downloads ~3.5 GB.
USE_MODELSCOPE = False
MODELSCOPE_REPO = "damo-vilab/text-to-video-ms-1.7b"

# ── ElevenLabs settings ───────────────────────────────────────────────────────
# Default = deep authoritative male ("Adam"). Override via .env if desired.
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")

# ── Local Ollama settings ─────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"          # Downloaded to D:\models\ollama
OLLAMA_MODELS_DIR = "D:/models/ollama"

# ── Local model storage (D: drive) ───────────────────────────────────────────
SDXL_CACHE_DIR = Path("D:/models/sd")    # SDXL weights cached here

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
CHARACTER_REFS_DIR = ASSETS_DIR / "character_refs"
STYLE_REFS_DIR = ASSETS_DIR / "style_refs"
MUSIC_DIR = ASSETS_DIR / "music"
STORIES_FILE = BASE_DIR / "stories" / "templates.json"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Clean single-subject crop of IMG_5950-edited.jpg (original had a 2nd person at
# the right edge that intruded into the SadTalker talking-head shots).
CHARACTER_REF_IMAGE = CHARACTER_REFS_DIR / "bob_clean.jpg"

# ── Models ────────────────────────────────────────────────────────────────────
GEMINI_TEXT_MODEL = "gemini-1.5-flash"      # Kept as fallback
GEMINI_IMAGE_MODEL = "imagen-3.0-generate-002"

# ── fal.ai model ids ──────────────────────────────────────────────────────────
FAL_KLING_MODEL = "fal-ai/kling-video/v2.1/standard/image-to-video"  # action-scene motion
FAL_FLUX_MODEL  = "fal-ai/flux/dev"                                  # still images
FAL_MUSIC_MODEL = "fal-ai/stable-audio"                              # cinematic music bed
FAL_LTX_MODEL   = "fal-ai/ltx-video/image-to-video"                 # cheaper i2v alternative
REPLICATE_COGVIDEO_MODEL = "zsxkib/cogvideox-5b:34702c4a6f28ab9bf79cc0d11ec63f0b0af63494f4c41f3e3f4f8da3af4f4c59"

# ── Video / image settings ────────────────────────────────────────────────────
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30
VIDEO_ASPECT = "9:16"
CLIP_DURATION_TARGET = 9
TOTAL_REEL_MIN_DURATION = 60

# ── Audio settings ────────────────────────────────────────────────────────────
TTS_VOICES = {
    "Guy (warm male)": "en-US-GuyNeural",
    "Davis (authoritative male)": "en-US-DavisNeural",
    "Aria (professional female)": "en-US-AriaNeural",
    "Jenny (friendly female)": "en-US-JennyNeural",
}
DEFAULT_TTS_VOICE = "en-US-GuyNeural"
SCENE_SPEECH_RATE = "-5%"   # edge-tts pace; slightly slower = more gravity
MUSIC_VOLUME_DB = -22

# ── Per-emotion color grade (the cold→warm narrative arc) ─────────────────────
# (r_mult, g_mult, b_mult) applied to every frame regardless of motion engine.
COLOR_GRADES = {
    "tension":     (1.05, 1.00, 0.90),   # warm amber  — the hook
    "shock":       (0.90, 0.92, 1.10),   # cold blue   — the crime
    "dramatic":    (0.85, 0.85, 0.95),   # desat grey  — lonely defeat
    "revelation":  (1.00, 0.97, 0.88),   # contrast gold — the climax line
    "triumph":     (1.08, 1.02, 0.92),   # full warm gold — redemption
    # legacy/spare moods kept for stories 2-8
    "connection":  (1.00, 1.00, 1.00),
    "curiosity":   (0.98, 0.98, 1.04),
    "neutral":     (1.00, 1.00, 1.00),
}

# ── Visual style presets ──────────────────────────────────────────────────────
STYLE_PRESETS = {
    "pixar_3d": {
        "label": "Pixar 3D",
        "description": "Warm Pixar-style 3D CGI animation, expressive character, soft lighting",
        "prompt_prefix": "Pixar-style 3D CGI illustration, warm soft lighting, expressive character design, vibrant saturated colors, cinematic composition",
    },
    "comic_book": {
        "label": "Comic Book",
        "description": "Bold graphic novel panels, strong outlines, flat bold colors",
        "prompt_prefix": "Bold comic book illustration, strong black outlines, flat vibrant colors, dynamic perspective, graphic novel style",
    },
    "cinematic": {
        "label": "Cinematic Dark",
        "description": "Dark moody cinematic, dramatic shadows, film grain",
        # NOTE: no "film grain"/"grainy"/"desaturated" here — weak SD 1.5 turns those
        # into literal grain/texture images. Grain + mood are applied in post
        # (assembler FX + COLOR_GRADES). Keep the prefix concrete and photographic.
        "prompt_prefix": "Cinematic photorealistic photograph, dramatic moody lighting, deep shadows, rich detail",
    },
    "flat_motion": {
        "label": "Flat Motion Graphics",
        "description": "Clean flat design, minimal illustration, modern brand feel",
        "prompt_prefix": "Flat design illustration, minimal clean style, bold geometric shapes, modern color palette",
    },
}
DEFAULT_STYLE = "pixar_3d"

# ── Character description (from reference images) ─────────────────────────────
CHARACTER_DESCRIPTION = (
    "a man in his mid-50s to early 60s, silver-gray hair, round glasses with thin frames, "
    "warm friendly smile, slightly weathered face with kind eyes, professional yet approachable"
)

# ── Tone options ──────────────────────────────────────────────────────────────
TONE_OPTIONS = {
    "dramatic": "High-stakes, emotional, cinematic narration with tension and payoff",
    "educational": "Calm, insightful, mentor-style wisdom sharing",
    "motivational": "Energetic, punchy, inspiring — designed to drive action",
}
DEFAULT_TONE = "dramatic"
