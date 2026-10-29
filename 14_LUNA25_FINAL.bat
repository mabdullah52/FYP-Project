@echo off
title STEP 14 - LUNA25 final results + cancer-risk model for the app (about 2-3 hours)
cd /d "%~dp0"
echo ============================================================
echo  STEP 14: 2 more training runs (different random seeds), then the final
echo  analysis: seed average, ensemble, size + age fusion, significance tests,
echo  uncertainty check, follow-up-scan check, and the cancer-risk model
echo  that the doctor app will use.
echo  If it stops, double-click again - it continues where it stopped.
echo ============================================================
set PY=lungenv\Scripts\python.exe run_logged.py 14_luna25_final
lungenv\Scripts\python.exe -m pip install psutil -q >nul 2>&1

call :trainseed 1
if errorlevel 1 goto failed
call :trainseed 2
if errorlevel 1 goto failed

%PY% luna25_final.py
if errorlevel 1 goto failed
echo.
echo SEND  logs\14_luna25_final.txt  and  work\luna25_final\final_table.csv
echo Then run 7_DOCTOR_APP.bat - it now also shows a cancer risk per nodule.
pause
exit /b

:trainseed
set N=0
:again
%PY% luna25_malignancy.py train --name l25_scratch_s%1 --seed %1
if not errorlevel 1 exit /b 0
set /a N+=1
if %N% GEQ 11 exit /b 1
echo Training stopped - restarting from the last saved epoch (try %N% of 10) ...
timeout /t 20 >nul
goto again

:failed
echo.
echo Something failed. SEND  logs\14_luna25_final.txt
pause
