@echo off
title STEP 19 - check the FAST doctor-app analysis on the 88 test scans (about 30-60 min)
cd /d "%~dp0"
echo Runs the faster nodule finder (GPU) + fast AI scoring on the same 88 test scans as step 17
echo and compares the score with the slow version (CPM 0.867). The time per scan is printed.
echo If it stops, double-click again - finished scans are skipped.
lungenv\Scripts\python.exe run_logged.py 19_speed_check eval_end_to_end.py --fast
echo.
echo SEND  logs\19_speed_check.txt
pause
