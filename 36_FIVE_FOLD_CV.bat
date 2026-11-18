@echo off
title STEP 36 - 5-fold cross-validation of MSGF-Net (about 2 days, resumable)
cd /d "%~dp0"
echo ============================================================
echo  STEP 36: 5-fold cross-validation for Paper 1 (MSGF-Net)
echo.
echo  Trains 5 new models. Each one is tested on 2 LUNA16 subsets it never saw,
echo  so all 888 scans are tested once. About 7 hours per fold, about 35 hours in total.
echo.
echo  First a 10-minute check runs. If it says SMOKE TEST PASSED, the long run starts.
echo  If it stops or the PC restarts, double-click this file again: it continues
echo  where it stopped and never redoes a finished fold.
echo.
echo  Do NOT click inside this window (it pauses the program).
echo  Do NOT run the doctor app or other GPU programs at the same time.
echo ============================================================
set PY=lungenv\Scripts\python.exe run_logged.py 36_cv5_msgf
call :run cv5.py run --arch msgf --smoke
if errorlevel 1 goto failed
call :run cv5.py run --arch msgf
if errorlevel 1 goto failed
%PY% cv5.py report
echo.
echo ============================================================
echo  FINISHED. SEND these files:
echo    logs\36_cv5_msgf.txt
echo    work\cv5\cv5_summary.csv
echo    work\cv5\cv5_folds.csv
echo    work\cv5\froc_cv5.png
echo ============================================================
pause
exit /b

:run
set N=0
:again
%PY% %*
if not errorlevel 1 exit /b 0
set /a N+=1
if %N% GEQ 11 exit /b 1
echo Stopped - continuing (try %N% of 10) ...
timeout /t 20 >nul
goto again

:failed
echo Something failed. SEND  logs\36_cv5_msgf.txt
pause
