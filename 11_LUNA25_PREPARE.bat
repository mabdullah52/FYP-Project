@echo off
title STEP 11 - LUNA25 preparation (1-2 hours, reads 214 GB)
cd /d "%~dp0"
echo ============================================================
echo  STEP 11: LUNA25 - read labels, patient-level split, cut nodule patches
echo  Needs LUNA25_Public_Training_Development_Data.csv (zenodo.org/records/14673658)
echo  If the PC restarts, double-click again - finished scans are skipped.
echo ============================================================
lungenv\Scripts\python.exe run_logged.py 11_luna25_prepare luna25_prepare.py --workers 8
echo.
echo SEND  logs\11_luna25_prepare.txt
pause
