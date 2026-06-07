@echo off
title RTX 5070 Video Generator
color 0A

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
    echo  Download Python 3.10+ from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during install.
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
if errorlevel 1 (
    echo  [....] PyTorch not found. Installing with CUDA 12.4 support.
    echo  [....] This is a large download (~2.5 GB). Please wait...
    echo  ------------------------------------------------
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 (
        echo.
        echo  [FAIL] PyTorch installation failed.
        pause
        exit /b 1
    )
    echo  ------------------------------------------------
    echo  [OK]   PyTorch installed successfully.
) else (
    for /f "delims=" %%v in ('python -c "import torch; print(torch.__version__)"') do set TORCHVER=%%v
    echo  [OK]   PyTorch %TORCHVER% already installed, skipping.
)
echo.

REM ── STEP 4: Other dependencies ────────────────────
echo  [STEP 4/5] Checking remaining dependencies...

python -c "import diffusers" >nul 2>&1
if errorlevel 1 (
    echo  [....] diffusers not found - will install from requirements.txt
    goto :install_reqs
)
python -c "import gradio" >nul 2>&1
if errorlevel 1 (
    echo  [....] gradio not found - will install from requirements.txt
    goto :install_reqs
)
python -c "import transformers" >nul 2>&1
if errorlevel 1 (
    echo  [....] transformers not found - will install from requirements.txt
    goto :install_reqs
)
python -c "import accelerate" >nul 2>&1
if errorlevel 1 (
    echo  [....] accelerate not found - will install from requirements.txt
    goto :install_reqs
)
echo  [OK]   All packages present, skipping install.
goto :after_reqs

:install_reqs
echo  [....] Installing packages from requirements.txt...
echo  ------------------------------------------------
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  [FAIL] Dependency installation failed.
    pause
    exit /b 1
)
echo  ------------------------------------------------
echo  [OK]   All packages installed.

:after_reqs
echo.

REM ── STEP 5: GPU check ─────────────────────────────
echo  [STEP 5/5] Detecting GPU...
python -c "import torch; cuda=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if cuda else 'NONE'; vram=round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if cuda else 0; status='[OK]  ' if cuda else '[WARN]'; print(f'  {status} GPU: {name}'); print(f'         VRAM: {vram} GB') if cuda else print('         WARNING: No CUDA GPU found. Inference will be very slow.')"
echo.

REM ── Launch ────────────────────────────────────────
echo  ================================================
echo   All checks passed. Launching app...
echo   URL: http://localhost:7860
echo   NOTE: First generation downloads the model (~8 GB)
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
