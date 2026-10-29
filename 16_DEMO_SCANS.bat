@echo off
title STEP 16 - which scans to use for the app demo
cd /d "%~dp0"
lungenv\Scripts\python.exe run_logged.py 16_demo_scans pick_demo_scans.py 1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260
echo.
echo Copy one of the paths above into the doctor app ("Add scan from a path on this PC").
echo The list is also saved in logs\16_demo_scans.txt
pause
