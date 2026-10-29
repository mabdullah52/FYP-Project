@echo off
title STEP 15 - Growth-aware cancer model for paper 2 (overnight, 4-7 hours)
cd /d "%~dp0"
echo ============================================================
echo  STEP 15 (paper 2): cancer model that compares each nodule with the
echo  SAME nodule on the previous year's screening scan.
echo   1) build current/prior pairs   2) 3 training runs of the proposed model
echo   3) 3 runs of the "no prior scan" control   4) final comparison table
echo  Leave it overnight. If it stops, double-click again - it continues.
echo ============================================================
set PY=lungenv\Scripts\python.exe run_logged.py 15_growth_model
%PY% luna25_temporal.py pairs
if errorlevel 1 goto failed
call :run 0 ""
if errorlevel 1 goto failed
call :run 1 ""
if errorlevel 1 goto failed
call :run 2 ""
if errorlevel 1 goto failed
call :run 0 --no-prior
if errorlevel 1 goto failed
call :run 1 --no-prior
if errorlevel 1 goto failed
call :run 2 --no-prior
if errorlevel 1 goto failed
%PY% luna25_temporal.py eval
if errorlevel 1 goto failed
echo.
echo SEND  logs\15_growth_model.txt  and  work\luna25_temporal\temporal_table.csv
pause
exit /b

:run
set N=0
:again
if "%~2"=="" (%PY% luna25_temporal.py train --seed %1) else (%PY% luna25_temporal.py train --seed %1 %~2)
if not errorlevel 1 exit /b 0
set /a N+=1
if %N% GEQ 11 exit /b 1
echo Training stopped - continuing from the last saved epoch (try %N% of 10) ...
timeout /t 20 >nul
goto again

:failed
echo.
echo Something failed. SEND  logs\15_growth_model.txt
pause
