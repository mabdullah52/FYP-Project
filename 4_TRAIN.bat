@echo off
title STEP 4 - Training (do not close, takes hours)
cd /d "%~dp0"
echo ============================================================
echo  STEP 4: TRAINING the model (several hours - leave the PC on).
echo  If it stops, double-click this file again - it continues.
echo ============================================================
lungenv\Scripts\python.exe run_logged.py 4_train train.py --name msgf_f9 --workers 8 --resume
echo.
echo SEND the files  logs\4_train.txt  and  work\runs\msgf_f9\history.csv
pause
