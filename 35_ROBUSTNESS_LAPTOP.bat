@echo off
title STEP 35 - robustness test (noisy and incomplete CT) on this laptop
cd /d "%~dp0"
echo ============================================================
echo  STEP 35: robustness test on the 88 LUNA16 test scans (subset 9)
echo  If it stops, double-click again: finished parts are skipped.
echo ============================================================
echo.
set DATA=E:\FYP\LUNA16
set LUNA16_ROOT=%DATA%
set PY=lungenv\Scripts\python.exe
set Z1=https://zenodo.org/records/3723295/files
set Z2=https://zenodo.org/records/4121926/files

rem ---------- 1. Python + packages ----------
if not exist %PY% python -m venv lungenv
%PY% -c "import torch, SimpleITK, scipy, pandas, sklearn, matplotlib" 2>nul
if errorlevel 1 (
  %PY% -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  %PY% -m pip install SimpleITK numpy scipy pandas scikit-learn matplotlib psutil
)
%PY% -c "import torch, SimpleITK, scipy, pandas, sklearn, matplotlib"
if errorlevel 1 (
  echo INSTALL FAILED - send a screenshot to Claude.
  pause
  exit /b 1
)
echo [1/3] Python packages OK.

rem ---------- 2. Data (downloads use -f so an error page is never written into a zip) ----------
if not exist "%DATA%" mkdir "%DATA%"
echo [2/3] LUNA16 data in %DATA%
if not exist "%DATA%\annotations.csv" curl.exe -f -L --retry 5 -o "%DATA%\annotations.csv" "%Z1%/annotations.csv?download=1"
if not exist "%DATA%\evaluationScript.zip" curl.exe -f -L --retry 5 -o "%DATA%\evaluationScript.zip" "%Z1%/evaluationScript.zip?download=1"
if not exist "%DATA%\candidates_V2.zip" curl.exe -f -L --retry 5 -o "%DATA%\candidates_V2.zip" "%Z1%/candidates_V2.zip?download=1"
if not exist "%DATA%\subset9\*.mhd" (
  echo Downloading or finishing subset9.zip - 6.7 GB ...
  curl.exe -f -L --retry 10 -C - -o "%DATA%\subset9.zip" "%Z2%/subset9.zip?download=1"
)
echo Unpacking with Python (works with large zips) ...
%PY% -c "import zipfile,os,sys;d=r'%DATA%';[ (print('  unpacking',z), zipfile.ZipFile(os.path.join(d,z)).extractall(d)) for z,chk in [('candidates_V2.zip','candidates_V2'),('evaluationScript.zip','evaluationScript'),('subset9.zip','subset9')] if os.path.exists(os.path.join(d,z)) and not os.path.isdir(os.path.join(d,chk))]"
if errorlevel 1 (
  echo.
  echo UNPACKING FAILED - the zip is damaged. Delete E:\FYP\LUNA16\subset9.zip
  echo and E:\FYP\LUNA16\candidates_V2.zip, then double-click this file again.
  pause
  exit /b 1
)
set NSCANS=0
for /f %%n in ('dir /b "%DATA%\subset9\*.mhd" 2^>nul ^| find /c /v ""') do set NSCANS=%%n
echo Found %NSCANS% test scans (should be 88).
if %NSCANS% LSS 88 (
  echo Not all scans are there. Delete E:\FYP\LUNA16\subset9 and subset9.zip and run again.
  pause
  exit /b 1
)

rem ---------- 3. Robustness test ----------
echo [3/3] Running the robustness test. The window shows how many minutes are left.
%PY% run_logged.py 35_robustness robustness.py
echo.
echo  DONE. Send Claude: logs\35_robustness.txt and work\robustness\robustness.png
pause
