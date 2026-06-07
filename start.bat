@echo off
setlocal enabledelayedexpansion
title RTX 5070 Video Generator
color 0A

REM Always run from the folder that contains this script
cd /d "%~dp0"

echo.
echo  ================================================
echo   RTX 5070 Video Generator
echo  ================================================
echo.

REM ── STEP 1: Check Python ──────────────────────────
echo  [STEP 1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo  [FAIL] Python not found.
    echo         Download Python 3.10+ from https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK]   Python %PYVER% detected.
echo.

REM ── STEP 2: Virtual environment ───────────────────
echo  [STEP 2/5] Setting up virtual environment...
if not exist ".venv\Scripts\activate.bat" (
    echo  [....] Creating new virtual environment in .venv\
    python -m venv .venv
    if errorlevel 1 (
        echo  [FAIL] Could not create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK]   Virtual environment created.
) else (
    echo  [OK]   Virtual environment already exists, skipping.
)
call .venv\Scripts\activate.bat
echo  [OK]   Virtual environment activated.
echo.

REM ── STEP 3: PyTorch ───────────────────────────────
echo  [STEP 3/5] Checking PyTorch + CUDA...
python -c "import torch" >nul 2>&1
if errorlevel 1 goto :install_torch
for /f "delims=" %%v in ('python -c "import torch; print(torch.__version__)"') do set TORCHVER=%%v
echo  [OK]   PyTorch %TORCHVER% already installed, skipping.
goto :torch_done

:install_torch
echo  [....] PyTorch not found. Installing with CUDA 12.8 support (RTX 5070).
echo  [....] This is a large download (~2.5 GB). Please wait...
echo  ------------------------------------------------
echo  [....] Pre-installing typing-extensions (fixes PyTorch index naming bug)...
pip install "typing-extensions>=4.10.0" --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo  [FAIL] PyTorch installation failed.
    pause
    exit /b 1
)
echo  ------------------------------------------------
echo  [OK]   PyTorch installed successfully.

:torch_done
echo.

REM ── STEP 4: Other dependencies ────────────────────
echo  [STEP 4/5] Checking remaining dependencies...
python -c "import diffusers" >nul 2>&1
if errorlevel 1 goto :install_reqs
python -c "import gradio" >nul 2>&1
if errorlevel 1 goto :install_reqs
python -c "import transformers" >nul 2>&1
if errorlevel 1 goto :install_reqs
python -c "import accelerate" >nul 2>&1
if errorlevel 1 goto :install_reqs
python -c "import tiktoken" >nul 2>&1
if errorlevel 1 goto :install_reqs
python -c "import google.protobuf" >nul 2>&1
if errorlevel 1 goto :install_reqs
echo  [OK]   All packages present, skipping install.
goto :reqs_done

:install_reqs
echo  [....] Missing packages detected. Installing from requirements.txt...
echo  ------------------------------------------------
pip install -r requirements.txt
if errorlevel 1 (
    echo  [FAIL] Dependency installation failed.
    pause
    exit /b 1
)
echo  ------------------------------------------------
echo  [OK]   All packages installed.

:reqs_done
echo.

REM ── STEP 5: GPU check ─────────────────────────────
echo  [STEP 5/5] Detecting GPU...
python -c "import torch; print('[OK]   GPU: ' + torch.cuda.get_device_name(0))" 2>nul
if errorlevel 1 echo  [WARN]  No CUDA GPU detected. Inference will be very slow.
python -c "import torch; print('       VRAM: ' + str(round(torch.cuda.get_device_properties(0).total_memory/1e9,1)) + ' GB')" 2>nul
echo.

REM ── Launch ────────────────────────────────────────
echo  ================================================
echo   All checks passed. Launching app...
echo   URL: http://localhost:7860
echo   NOTE: First generation downloads the model (~8 GB).
echo   Press Ctrl+C in this window to stop the server.
echo  ================================================
echo.

start "" /B cmd /C "timeout /t 4 >nul && start http://localhost:7860"
python app.py

echo.
echo  ================================================
echo   App has stopped.
echo  ================================================
pause
