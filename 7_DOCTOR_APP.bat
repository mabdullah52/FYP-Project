@echo off
title LungScan AI - Doctor app (keep this window open)
cd /d "%~dp0"
lungenv\Scripts\python.exe -c "import fastapi, uvicorn, multipart, PIL" >nul 2>nul
if errorlevel 1 (
  echo Installing the web app packages once - needs internet...
  lungenv\Scripts\python.exe -m pip install fastapi uvicorn python-multipart pillow
)
rem listen on the network too, so the Android doctor/patient apps on the same Wi-Fi can connect
set LUNGSCAN_HOST=0.0.0.0
set URL=http://127.0.0.1:8000
if exist doctor_flutter\build\web\index.html set URL=http://127.0.0.1:8000/doctor/
echo.
echo  Doctor app is starting. Your browser will open at %URL%
echo  Keep this window open while you use the app. Close it to stop the app.
echo  (If Windows Firewall asks, allow access on private networks.)
echo.
start "" %URL%
lungenv\Scripts\python.exe doctor_app\server.py
pause
