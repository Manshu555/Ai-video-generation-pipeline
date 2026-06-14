# Breakout AI — Viral Reel Generator 🎬

Breakout AI is an automated, multi-step pipeline for generating 60-second vertical video reels (9:16) from text stories. It is designed to take a narrative (or select from pre-written business stories) and turn it into a production-ready video with voiceovers, animated scenes, music, and captions.

The system is highly modular, prioritizing local generation (using tools like Ollama, SDXL, and SadTalker) while supporting cloud-based providers (fal.ai, ElevenLabs, Hedra) for higher-quality or faster results.

## Key Features

- **End-to-End Pipeline**: Transforms a story into a complete video in 8 steps:
  *Story → Style → Script → Images → Clips → Audio → Assemble → Export*
- **Character Consistency**: Preserves the identity of a specific character across different scenes and styles using multi-layered prompting and reference images.
- **Multiple Visual Styles**: Supports "Pixar 3D", "Comic Book", "Cinematic Dark", and "Flat Motion Graphics".
- **Dynamic Animation**: Intelligently routes speaking scenes to lip-sync engines (SadTalker/Hedra) and action scenes to motion engines (AnimateDiff/Kling/Ken Burns).
- **Flexible Infrastructure**: Operates entirely locally on a capable GPU (RTX 4060+) or utilizes cloud APIs (`fal.ai`, ElevenLabs) when keys are provided.
- **Dual Interface**: Run via a Streamlit web UI wizard (`app.py`) or headlessly via script (`generate_video.py`).

## Tech Stack & Supported Engines

*   **Script Generation**: Local Ollama (e.g., `llama3.2:3b`), Gemini Flash
*   **Image Generation**: Local SDXL / SD 1.5, `fal.ai` Flux/Ideogram, Gemini Imagen 3
*   **Video Animation (Action)**: Local AnimateDiff, `fal.ai` Kling, Replicate CogVideoX, Local MoviePy Ken Burns (fallback)
*   **Video Animation (Speaking)**: Local SadTalker, Hedra
*   **Audio Generation**: `edge-tts` (free/local-ish), ElevenLabs
*   **Video Assembly**: MoviePy v2 (transitions, text overlays, captions, background music)

## Setup & Installation

1.  **Clone the Repository** and open the project directory.
2.  **Environment Setup**:
    Run the provided setup script to create a virtual environment and install dependencies:
    ```bash
    setup_venv.bat
    ```
    *(Alternatively, manually create a venv and run `pip install -r requirements.txt`)*
3.  **Local Models (Optional but recommended)**:
    - Download and install [Ollama](https://ollama.com/) and pull the required model (e.g., `ollama run llama3.2:3b`).
    - Configure HuggingFace caches and local paths in `config.py` if needed.
4.  **Environment Variables**:
    Copy `.env.example` to `.env` (if available) and configure your API keys to enable cloud providers:
    ```env
    FAL_KEY=your_fal_key
    ELEVENLABS_API_KEY=your_elevenlabs_key
    HEDRA_API_KEY=your_hedra_key
    ```
    *Note: If no keys are provided, the pipeline defaults to local engines.*

## Usage

### 1. Using the Streamlit UI (Recommended)

Start the interactive 8-step wizard:

```bash
run_app.bat
```
*(Or manually: `streamlit run app.py`)*

The UI guides you through:
1. Selecting a pre-written viral story or writing your own.
2. Choosing a visual style and tone.
3. Reviewing and editing the generated script.
4. Generating base images.
5. Animating the scenes.
6. Generating voiceovers and selecting music.
7. Assembling the final video.
8. Exporting and uploading to Google Drive.

### 2. Using the CLI Script

For headless generation or bulk production, edit the constants in `generate_video.py` (Story ID, Style, Tone, Voice) and run:

```bash
python generate_video.py
```

The final assembled MP4 will be saved to `output.mp4` in the project root, and intermediate files will be saved in the `outputs/session_{timestamp}/` directory.

## Project Structure

*   `app.py` - Streamlit application entry point.
*   `generate_video.py` - CLI script for end-to-end headless video generation.
*   `config.py` - Global configuration, model paths, API toggles, and style presets.
*   `pipeline/` - Core modules for each generation step:
    *   `script_generator.py` - LLM interaction for scene structuring.
    *   `image_generator.py` - T2I generation.
    *   `video_generator.py` - Animation and lip-syncing.
    *   `audio_generator.py` - TTS and music handling.
    *   `assembler.py` - Final video composition via MoviePy.
    *   `providers/` - API integrations (fal, Hedra, ElevenLabs).
*   `assets/` - Contains character reference images, style references, and music.
*   `outputs/` - Generated scripts, images, clips, audio, and final reels.
*   `stories/` - JSON templates for pre-written viral stories.

## Workflow Strategy

The system relies on a multi-modal strategy to ensure quality and consistency:
1.  **Strict Script Formatting**: The LLM output is enforced as JSON, separating visual descriptions, voiceovers, and camera motion for reliable downstream processing.
2.  **Face Consistency**: The prompt injects a locked textual description of the character, combined with reference images (where the generator supports it), to keep the subject recognizable across generated scenes.
3.  **Shot Routing**: Scenes tagged as "speaking" use lip-sync AI (Hedra/SadTalker) on portrait crops. Scenes tagged as "action" use generic motion AI (Kling/AnimateDiff) or Ken Burns effects to create dynamic B-roll.
