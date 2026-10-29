@echo off
title STEP 10 - LUNA25 check + significance tests (a few minutes, no GPU)
cd /d "%~dp0"
echo ============================================================
echo  STEP 10: find + inspect the LUNA25 folder, and paper statistics
echo  Tip: you can also DRAG your LUNA25 folder onto this .bat file.
echo ============================================================
if "%~1"=="" (
  lungenv\Scripts\python.exe run_logged.py 10_luna25_and_stats luna25_audit.py
) else (
  lungenv\Scripts\python.exe run_logged.py 10_luna25_and_stats luna25_audit.py "%~1"
)
lungenv\Scripts\python.exe run_logged.py 10_luna25_and_stats stats_compare.py
echo.
echo SEND  logs\10_luna25_and_stats.txt
pause
