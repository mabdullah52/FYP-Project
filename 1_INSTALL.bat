@echo off
title STEP 1 - Install
cd /d "%~dp0"
echo ============================================================
echo  STEP 1: installing Python packages (needs internet, 10-20 min)
echo ============================================================
where python >nul 2>nul
if errorlevel 1 (
  echo Python is not installed. Install Python 3.11 from https://www.python.org/downloads/
  echo During install TICK the box "Add python.exe to PATH". Then double-click this file again.
  pause
  exit /b 1
)
if not exist lungenv\Scripts\python.exe python -m venv lungenv
if not exist lungenv\Scripts\python.exe (
  echo Could not create the Python environment. Install Python 3.11 from https://www.python.org/downloads/
  echo and TICK "Add python.exe to PATH". Then double-click this file again.
  pause
  exit /b 1
)
lungenv\Scripts\python.exe -m pip install --upgrade pip
lungenv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto fail
lungenv\Scripts\python.exe -m pip install SimpleITK numpy scipy pandas scikit-learn matplotlib psutil
if errorlevel 1 goto fail
echo.
echo DONE. Now double-click 2_CHECK.bat
pause
exit /b 0
:fail
echo.
echo INSTALL FAILED - take a photo/screenshot of this window and send it.
pause
exit /b 1
