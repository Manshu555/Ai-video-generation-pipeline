@echo off
echo === Installing remaining packages into D:\round2_venv ===
set PIP_CACHE_DIR=D:\pip_cache

D:\round2_venv\Scripts\pip.exe install ^
    diffusers>=0.30.0 ^
    transformers>=4.40.0 ^
    accelerate>=0.30.0 ^
    safetensors>=0.4.0 ^
    moviepy>=2.0.0 ^
    streamlit>=1.35.0 ^
    edge-tts>=7.0.0 ^
    pydub>=0.25.1 ^
    requests>=2.31.0 ^
    python-dotenv>=1.0.0 ^
    ollama ^
    Pillow>=10.0.0 ^
    numpy ^
    imageio ^
    imageio-ffmpeg ^
    proglog

echo === Done! Run run_app.bat to start the app ===
pause
