@echo off
title STEP 30 - figures for paper 2 (about 2 minutes)
cd /d "%~dp0"
lungenv\Scripts\python.exe run_logged.py 30_paper2_figures paper2_figures.py
echo.
echo SEND  the 3 pictures in work\paper2_figures  (roc_single.png, auc_by_size.png, roc_temporal.png)
echo       and logs\30_paper2_figures.txt
pause
