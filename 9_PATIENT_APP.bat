@echo off
title LungCare - Patient app (keep this window open)
cd /d "%~dp0"
lungenv\Scripts\python.exe -c "import fastapi, uvicorn, anthropic, pypdf" >nul 2>nul
if errorlevel 1 (
  echo Installing packages once - needs internet...
  lungenv\Scripts\python.exe -m pip install fastapi uvicorn python-multipart pillow anthropic pypdf
)
lungenv\Scripts\python.exe -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" >nul 2>nul
if errorlevel 1 (
  echo The doctor app is not running - starting it in a second window...
  start "LungScan AI - Doctor app" cmd /k lungenv\Scripts\python.exe doctor_app\server.py
  timeout /t 8 >nul
)
echo.
echo  Patient app: http://127.0.0.1:8100   (log in with the code shown in the doctor app)
echo  Android app: use the address shown below (same Wi-Fi). Official guideline PDFs can be added to patient_app\knowledge
echo  For the Claude AI assistant, paste your API key into patient_app\claude_api_key.txt
echo  and restart this window. Without a key it runs in offline mode.
echo.
start "" http://127.0.0.1:8100
set LUNGCARE_HOST=0.0.0.0
lungenv\Scripts\python.exe patient_app\server.py
pause
