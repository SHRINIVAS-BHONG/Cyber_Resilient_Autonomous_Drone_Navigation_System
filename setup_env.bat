@echo off
title Setup Python 3.10 Virtual Environment
color 0B
cd /d "%~dp0"

echo ===========================================================================
echo   PYTHON 3.10 ENVIRONMENT SETUP - CYBER-RESILIENT DRONE SYSTEM
echo ===========================================================================
echo.

set PYTHON_CMD=

:: 1. Check py launcher for Python 3.10
py -3.10 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.10
    goto found_py
)

:: 2. Check LocalAppData for Python 3.10
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    goto found_py
)

:: 3. Check C:\Python310
if exist "C:\Python310\python.exe" (
    set PYTHON_CMD="C:\Python310\python.exe"
    goto found_py
)

:: 4. Fallback check for default python
for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do (
    if "%%i"=="3.10" (
        set PYTHON_CMD=python
        goto found_py
    )
)

echo [-] ERROR: Python 3.10 was not found on your system!
echo.
echo     Please install Python 3.10 from:
echo     https://www.python.org/downloads/release/python-31011/
echo.
echo     Ensure you check the box: "Add Python 3.10 to PATH" during installation.
echo.
pause
exit /b 1

:found_py
echo [+] Found Python 3.10: %PYTHON_CMD%
echo.
echo [*] Creating virtual environment (.venv)...
%PYTHON_CMD% -m venv "%~dp0.venv"
if %errorlevel% neq 0 (
    echo [-] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [+] Virtual environment created successfully in .venv.
echo.
echo [*] Upgrading pip and installing Python 3.10 compatible dependencies...
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"

if %errorlevel% neq 0 (
    echo [-] Dependency installation encountered issues.
    pause
    exit /b 1
)

echo.
echo ===========================================================================
echo   SETUP COMPLETE! Python 3.10 environment is ready.
echo   You can now run START_SYSTEM.bat or execute:
echo     .venv\Scripts\python.exe run.py
echo ===========================================================================
echo.
pause
