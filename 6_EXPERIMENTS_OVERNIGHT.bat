@echo off
title STEP 6 - Ablation experiments for the paper (about 10-12 h)
cd /d "%~dp0"
echo ============================================================
echo  STEP 6: 6 comparison experiments for the paper (run overnight)
echo  If it stops, double-click again - finished parts are skipped.
echo ============================================================
set PY=lungenv\Scripts\python.exe run_logged.py 6_experiments
%PY% train.py --name abl_full      --epochs 10 --workers 8 --resume
%PY% train.py --name abl_nocontext --epochs 10 --workers 8 --resume --no-context
%PY% train.py --name abl_concat    --epochs 10 --workers 8 --resume --fusion concat
%PY% train.py --name abl_1window   --epochs 10 --workers 8 --resume --windows single
%PY% train.py --name abl_noaux     --epochs 10 --workers 8 --resume --aux-weight 0
%PY% train.py --name abl_bce       --epochs 10 --workers 8 --resume --loss bce
for %%R in (abl_full abl_nocontext abl_concat abl_1window abl_noaux abl_bce) do (
  if not exist work\runs\%%R\eval_subset9_plain\metrics.json %PY% evaluate.py --run %%R --no-tta --mc 1 --tag _plain --n-boot 200 --workers 8
)
%PY% collect_results.py
echo.
echo SEND  work\results_table.csv
pause
