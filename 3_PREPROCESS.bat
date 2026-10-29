@echo off
title STEP 3 - Preprocessing (do not close)
cd /d "%~dp0"
echo ============================================================
echo  STEP 3: PREPROCESSING the CT scans (30-90 min).
echo  If the PC restarts, double-click this file again - it continues.
echo ============================================================
lungenv\Scripts\python.exe run_logged.py 3_preprocess preprocess.py --workers 8
echo.
echo SEND the file  logs\3_preprocess.txt
pause
