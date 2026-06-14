@echo off
title Breakout AI - Setup and Generate Video
echo ============================================================
echo  Breakout AI Pipeline - Full Setup and Video Generation
echo ============================================================
echo.

set PIP_CACHE_DIR=D:\pip_cache
set KMP_DUPLICATE_LIB_OK=TRUE
set HF_HOME=D:\models\sd
set TRANSFORMERS_CACHE=D:\models\sd

echo [1/3] Installing required Python packages...
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

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Package installation failed!
    pause
    exit /b 1
)

echo.
echo [2/3] Verifying GPU and torch...
D:\round2_venv\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PyTorch CUDA not working!
    pause
    exit /b 1
)

echo.
echo [3/3] Generating video...
cd /d d:\round2_vs_code
D:\round2_venv\Scripts\python.exe generate_video.py

echo.
echo ============================================================
echo  Done! Check d:\round2_vs_code\output.mp4
echo ============================================================
pause
