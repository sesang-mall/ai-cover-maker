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

echo [3/4] Installing rvc-python (skipping broken deps)...
pip install rvc-python --no-deps
if errorlevel 1 (
    echo ERROR: rvc-python installation failed.
    pause & exit /b 1
)

echo [4/4] Installing remaining packages...
echo (This may take 10-20 minutes)
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
