@echo off
title STEP 20 - redraw the triage figure for the paper (a few seconds)
cd /d "%~dp0"
lungenv\Scripts\python.exe run_logged.py 20_triage replot_triage.py
echo.
echo SEND  work\runs\msgf_f9\eval_subset9\triage.png  and  logs\20_triage.txt
pause
