@echo off
title STEP 31 - 5-fold cross-validation for paper 2 (about 20-24 hours, resumable)
cd /d "%~dp0"
echo ============================================================
echo  STEP 31: trains the cancer-risk model and the growth model on 4 more patient groups,
echo  so EVERY LUNA25 nodule is tested once (fold 0 = the models you already have).
echo  Per group: single-scan model (~2-3 h) + growth model (~1-1.5 h); then the
echo  "no prior" control models; then the final report.
echo  If it stops or you close it, double-click again - it continues where it stopped.
echo  Do NOT click inside this window (it pauses the program).
echo ============================================================
set PY=lungenv\Scripts\python.exe run_logged.py 31_cross_validation
for %%k in (1 2 3 4) do (
  set L25_CV_FOLD=%%k
  call :run luna25_malignancy.py train --name l25_cv%%k
  if errorlevel 1 goto failed
  call :run luna25_temporal.py train --seed 0
  if errorlevel 1 goto failed
)
for %%k in (1 2 3 4) do (
  set L25_CV_FOLD=%%k
  call :run luna25_temporal.py train --seed 0 --no-prior
  if errorlevel 1 goto failed
)
set L25_CV_FOLD=
%PY% luna25_cv.py report
echo.
echo SEND  logs\31_cross_validation.txt  and  work\luna25_cv\cv_table.csv
pause
exit /b

:run
set N=0
:again
echo.
echo ---- group %L25_CV_FOLD%: %* ----
%PY% %*
if not errorlevel 1 exit /b 0
set /a N+=1
if %N% GEQ 11 exit /b 1
echo Stopped - continuing from the last saved epoch (try %N% of 10) ...
timeout /t 20 >nul
goto again

:failed
echo Something failed. SEND  logs\31_cross_validation.txt
pause
