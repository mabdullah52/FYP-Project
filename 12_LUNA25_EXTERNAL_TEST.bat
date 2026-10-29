@echo off
title STEP 12 - LUNA25 external test of the LUNA16 model (20-40 min)
cd /d "%~dp0"
lungenv\Scripts\python.exe run_logged.py 12_luna25_external luna25_external.py
echo.
echo SEND  logs\12_luna25_external.txt
pause
