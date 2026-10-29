@echo off
title STEP 34 - compare MSGF-Net with standard models (resumable)
cd /d "%~dp0"
echo ============================================================
echo  STEP 34: trains 3 standard models on exactly the same data as MSGF-Net
echo   - 3D ResNet-18   - 2.5D ResNet-18   - plain 3D CNN
echo  then tests each on the 88 test scans and makes the comparison table.
echo  If it stops, double-click again - it continues where it stopped.
echo  Do NOT click inside this window. Do not run the doctor app at the same time.
echo ============================================================
set PY=lungenv\Scripts\python.exe run_logged.py 34_baselines
for %%a in (cnn3d resnet25d) do (
  call :run train.py --name base_%%a --arch %%a --resume
  if errorlevel 1 goto failed
)
rem 3D ResNet-18 with the standard (fast) stem; the old slow run folder is removed once
if exist work\runs\base_resnet3d18 rmdir /s /q work\runs\base_resnet3d18
call :run train.py --name base_r3d18 --arch resnet3d18 --resume
if errorlevel 1 goto failed
for %%a in (cnn3d resnet25d r3d18) do (
  if not exist work\runs\base_%%a\eval_subset9\metrics.json (
    call :run evaluate.py --run base_%%a
    if errorlevel 1 goto failed
  )
)
%PY% compare_baselines.py
echo.
echo SEND  logs\34_baselines.txt  and  work\baseline_table.csv
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
echo Something failed. SEND  logs\34_baselines.txt
pause
