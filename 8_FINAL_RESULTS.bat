@echo off
title STEP 8 - Final result tables for the paper (about 3-4 hours)
cd /d "%~dp0"
echo ============================================================
echo  STEP 8: fair model selection + final tables (3-4 h)
echo   - round-1 model: full test + validation score
echo   - fast single-pass scores of round 1 and round 2 on validation and test
echo   - ensemble (round 1 + round 2)
echo   - one results table, with annotations_excluded.csv if it is present
echo  If it stops, double-click again - finished parts are skipped.
echo ============================================================
set PY=lungenv\Scripts\python.exe run_logged.py 8_final_results
if not exist work\runs\msgf_f9\eval_subset9\metrics.json %PY% evaluate.py --run msgf_f9 --calibrate --workers 8
for %%R in (msgf_f9 msgf_f9_r2) do (
  if not exist work\runs\%%R\eval_subset8_plain\metrics.json %PY% evaluate.py --run %%R --subset 8 --no-tta --mc 1 --tag _plain --n-boot 200 --workers 8
  if not exist work\runs\%%R\eval_subset9_plain\metrics.json %PY% evaluate.py --run %%R --no-tta --mc 1 --tag _plain --n-boot 200 --workers 8
)
%PY% ensemble.py --runs msgf_f9 msgf_f9_r2 --eval eval_subset8_plain --name ens_r1r2
%PY% ensemble.py --runs msgf_f9 msgf_f9_r2 --eval eval_subset9 --name ens_r1r2
%PY% collect_results.py
echo.
echo SEND  work\results_table.csv  and  logs\8_final_results.txt
pause
