@echo off
title STEP 3B - Check the preprocessed patches (1 min)
cd /d "%~dp0"
lungenv\Scripts\python.exe run_logged.py 3b_check_patches show_patches.py
echo.
echo SEND  logs\3b_check_patches.txt  and the picture  work\patch_check.png
pause
