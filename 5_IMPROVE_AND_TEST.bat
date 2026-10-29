@echo off
title STEP 5 - Round 2 + final test (do not close)
cd /d "%~dp0"
echo ============================================================
echo  STEP 5: hard-negative round 2 + final test on unseen patients (2-4 h)
echo ============================================================
if not exist work\patches\hard_msgf_f9.csv (
  lungenv\Scripts\python.exe run_logged.py 5_final mine_hard_negatives.py --run msgf_f9 --top-k 20 --workers 8
)
lungenv\Scripts\python.exe run_logged.py 5_final train.py --name msgf_f9_r2 --init-from msgf_f9 --hard-meta work\patches\hard_msgf_f9.csv --epochs 10 --lr 3e-4 --workers 8 --resume
lungenv\Scripts\python.exe run_logged.py 5_final evaluate.py --run msgf_f9_r2 --calibrate --workers 8
echo.
echo SEND  logs\5_final.txt  and the folder  work\runs\msgf_f9_r2\eval_subset9
pause
