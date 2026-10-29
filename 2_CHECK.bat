@echo off
title STEP 2 - Check computer and dataset
cd /d "%~dp0"
echo ============================================================
echo  STEP 2: checking GPU, packages and the LUNA16 folder (about 5 min)
echo ============================================================
lungenv\Scripts\python.exe run_logged.py 2_check check_setup.py
lungenv\Scripts\python.exe run_logged.py 2_check preprocess.py --audit
echo.
echo SEND the file  logs\2_check.txt  (it is inside this folder)
pause
