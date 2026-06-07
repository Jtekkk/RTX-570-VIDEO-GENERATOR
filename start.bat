@echo off
title RTX 5070 Video Generator
color 0A

echo.
echo  ================================================
echo   RTX 5070 Video Generator
echo  ================================================
echo.

REM ── Check Python ──────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    echo  Download Python 3.10+ from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Python %PYVER% found.

REM ── Create virtual environment ────────────────────
if not exist ".venv\Scripts\activate.bat" (
    echo  Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

REM ── Install / verify dependencies ─────────────────
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo  Installing PyTorch with CUDA 12.4 support...
    echo  (This may take several minutes on first run)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet
)

python -c "import gradio" >nul 2>&1
if errorlevel 1 (
    echo  Installing remaining dependencies...
    pip install -r requirements.txt --quiet
)

REM ── Check GPU ─────────────────────────────────────
echo.
python -c "import torch; print('  GPU: ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT DETECTED (CPU mode)'))"
python -c "import torch; vram=torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0; print(f'  VRAM: {vram:.1f} GB') if vram else None"
echo.

REM ── Launch ────────────────────────────────────────
echo  Starting app at http://localhost:7860
echo  Opening browser automatically...
echo.
echo  Press Ctrl+C to stop.
echo.

REM Open browser after a short delay
start "" /B cmd /C "timeout /t 3 >nul && start http://localhost:7860"

python app.py

echo.
echo  App stopped.
pause
