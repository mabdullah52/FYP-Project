@echo off
title STEP 15B - growth model: final comparison only (about 30-60 min, no training)
cd /d "%~dp0"
lungenv\Scripts\python.exe run_logged.py 15b_growth_eval luna25_temporal.py eval
echo.
echo SEND  logs\15b_growth_eval.txt
pause
