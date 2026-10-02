@echo off
title Cyber-Resilient Drone Navigation System
color 0A
cd /d "%~dp0"
echo ===========================================================================
echo   STARTING CYBER-RESILIENT AUTONOMOUS DRONE NAVIGATION SYSTEM (Python 3.10)
echo ===========================================================================
echo.

:: 1. Check local virtual environment (.venv)
if exist "%~dp0.venv\Scripts\python.exe" (
    echo [*] Using local Python 3.10 virtual environment: .venv
    "%~dp0.venv\Scripts\python.exe" run.py
    goto end
)

:: 2. Check py launcher for Python 3.10
py -3.10 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [*] Detected Python 3.10 via Windows py launcher.
    py -3.10 run.py
    goto end
)

:: 3. Check common Python 3.10 install directories
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    echo [*] Found Python 3.10 in LocalAppData.
    
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" run.py
    goto end
)
if exist "C:\Python310\python.exe" (
    echo [*] Found Python 3.10 in C:\Python310.
    "C:\Python310\python.exe" run.py
    goto end
)

:: 4. Fallback to default python
echo [!] Warning: Python 3.10 was not explicitly found in standard locations.
echo     Attempting default python command...
python run.py

:end
pause

