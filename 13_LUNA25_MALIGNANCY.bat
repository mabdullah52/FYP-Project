@echo off
title STEP 13 - LUNA25 malignancy model (3-5 hours, leave the PC on)
cd /d "%~dp0"
echo ============================================================
echo  STEP 13: benign vs malignant model on LUNA25
echo   1) transfer from the LUNA16 model (proposed)   2) from scratch (comparison)
echo   3) evaluation on the held-out patients + final table
echo  BEFORE STARTING: close the browser and any other Python windows.
echo  If it stops, double-click again - it continues from the last epoch.
echo ============================================================
set PY=lungenv\Scripts\python.exe run_logged.py 13_luna25_malignancy
lungenv\Scripts\python.exe -m pip install psutil -q >nul 2>&1

set N=0
:transfer
%PY% luna25_malignancy.py train --name l25_transfer --init msgf_f9
if not errorlevel 1 goto scratch_start
set /a N+=1
if %N% GEQ 11 goto failed
echo Training stopped - restarting from the last saved epoch (try %N% of 10) ...
timeout /t 20 >nul
goto transfer

:scratch_start
set N=0
:scratch
%PY% luna25_malignancy.py train --name l25_scratch
if not errorlevel 1 goto evaluate
set /a N+=1
if %N% GEQ 11 goto failed
echo Training stopped - restarting from the last saved epoch (try %N% of 10) ...
timeout /t 20 >nul
goto scratch

:evaluate
%PY% luna25_malignancy.py eval --name l25_transfer
%PY% luna25_malignancy.py eval --name l25_scratch
%PY% luna25_malignancy.py table
echo.
echo SEND  logs\13_luna25_malignancy.txt  and  work\luna25_malignancy_table.csv
pause
exit /b

:failed
echo.
echo Training failed 10 times. Restart the PC, close other programs, run this file again.
echo SEND  logs\13_luna25_malignancy.txt
pause
