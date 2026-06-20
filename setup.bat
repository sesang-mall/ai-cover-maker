@echo off
echo.
echo ==========================================
echo   AI Cover Maker - Setup
echo ==========================================
echo.

echo [1/3] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found.
    pause & exit /b 1
)

echo [2/3] Checking FFmpeg...
ffmpeg -version > nul 2>&1
if errorlevel 1 (
    echo WARNING: FFmpeg not found.
    echo   Install: winget install FFmpeg
    echo.
)

echo [3/3] Installing packages...
echo (This may take 10-20 minutes - rvc-python includes large models)
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Installation failed.
    pause & exit /b 1
)

echo.
echo ==========================================
echo   Setup complete!
echo   Run: python main.py
echo ==========================================
pause
