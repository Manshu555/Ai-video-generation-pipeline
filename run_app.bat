@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
set HF_HOME=D:\models\sd
set TRANSFORMERS_CACHE=D:\models\sd
set HF_HUB_DISABLE_PROGRESS_BARS=0
cd /d d:\round2_vs_code
D:\round2_venv\Scripts\streamlit.exe run app.py
