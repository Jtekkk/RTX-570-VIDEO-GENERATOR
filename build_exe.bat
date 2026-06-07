@echo off
title Build EXE - RTX 5070 Video Generator
color 0B

echo.
echo  ================================================
echo   Build Standalone EXE (PyInstaller)
echo  ================================================
echo.
echo  WARNING: The resulting EXE will be 3-5 GB due to PyTorch.
echo  It will still download the LTX-Video model (~8 GB) on first run.
echo  For most users, start.bat is the better option.
echo.
set /p CONFIRM="Continue building? (y/n): "
if /i not "%CONFIRM%"=="y" exit /b 0

REM ── Activate venv (run start.bat first to set it up) ──
if not exist ".venv\Scripts\activate.bat" (
    echo  [ERROR] Virtual environment not found. Run start.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

echo.
echo  Installing PyInstaller...
pip install pyinstaller --quiet

echo  Building EXE...
echo.

pyinstaller ^
    --onefile ^
    --name "RTX5070VideoGen" ^
    --icon NONE ^
    --collect-all gradio ^
    --collect-all gradio_client ^
    --hidden-import diffusers ^
    --hidden-import transformers ^
    --hidden-import accelerate ^
    --hidden-import torch ^
    --hidden-import imageio ^
    --hidden-import imageio_ffmpeg ^
    app.py

if errorlevel 1 (
    echo.
    echo  [ERROR] Build failed. Check the output above for details.
    pause
    exit /b 1
)

echo.
echo  ================================================
echo   Build complete!
echo   EXE location: dist\RTX5070VideoGen.exe
echo  ================================================
echo.
pause
