@echo off
title STEP 17 - full-system test on the 88 test scans (about 1-1.5 hours)
cd /d "%~dp0"
echo Tests the app pipeline exactly as it runs on a new scan: own nodule finder, then the model.
echo If it stops, double-click again - finished scans are skipped.
lungenv\Scripts\python.exe run_logged.py 17_end_to_end eval_end_to_end.py
echo.
echo SEND  logs\17_end_to_end.txt
pause
