@echo off
title STEP 37 - OPTIONAL 5-fold run of the plain 3D CNN (about 4 days, resumable)
cd /d "%~dp0"
echo ============================================================
echo  STEP 37 (OPTIONAL): the same 5 folds for the plain 3D CNN comparison model.
echo  Run this only AFTER step 36 has finished, and only if you decided to.
echo  About 19 hours per fold, about 4 days in total. Resumable like step 36.
echo  At the end it compares MSGF-Net and the plain 3D CNN on all 888 scans.
echo  Do NOT click inside this window. Do not run other GPU programs.
echo ============================================================
if not exist work\cv5\cv5_summary.csv (
  echo Step 36 has not finished yet. Run 36_FIVE_FOLD_CV.bat first.
  pause
  exit /b
)
set PY=lungenv\Scripts\python.exe run_logged.py 37_cv5_cnn3d
call :run cv5.py run --arch cnn3d
if errorlevel 1 goto failed
%PY% cv5.py report
echo.
echo SEND  logs\37_cv5_cnn3d.txt  work\cv5\cv5_summary.csv  work\cv5\cv5_comparison.csv  work\cv5\froc_cv5.png
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
echo Something failed. SEND  logs\37_cv5_cnn3d.txt
pause
