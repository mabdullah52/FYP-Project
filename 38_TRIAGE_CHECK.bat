@echo off
title STEP 38 - uncertainty check (about 1 minute)
cd /d "%~dp0"
lungenv\Scripts\python.exe run_logged.py 38_cv5_triage cv5.py triage
echo.
echo SEND a photo of this window, or the file  work\cv5\cv5_triage.csv
pause
