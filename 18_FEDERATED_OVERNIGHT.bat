@echo off
title STEP 18 - Swarm / federated learning experiment (overnight, about 8-10 hours)
cd /d "%~dp0"
echo ============================================================
echo  STEP 18: privacy-preserving training across 4 simulated hospitals.
echo   1) federated (FedAvg): hospitals share only model weights
echo   2) one hospital alone      3) all data pooled (upper bound)
echo   4) test all three on the 88 test scans + comparison table
echo  If it stops, double-click again - it continues where it stopped.
echo ============================================================
set PY=lungenv\Scripts\python.exe run_logged.py 18_federated
call :run fedavg
if errorlevel 1 goto failed
call :run local --site A
if errorlevel 1 goto failed
call :run central
if errorlevel 1 goto failed
%PY% fed_train.py report
echo.
echo SEND  logs\18_federated.txt  and  work\federated_table.csv
pause
exit /b

:run
set N=0
:again
%PY% fed_train.py %*
if not errorlevel 1 exit /b 0
set /a N+=1
if %N% GEQ 11 exit /b 1
echo Stopped - continuing from the last saved round (try %N% of 10) ...
timeout /t 20 >nul
goto again

:failed
echo Something failed. SEND  logs\18_federated.txt
pause
