@echo off
title STEP 1B - reinstall PyTorch for older GPU drivers
cd /d "%~dp0"
echo Use this ONLY if 2_CHECK said the GPU failed. Reinstalling PyTorch (CUDA 11.8 build)...
lungenv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
echo DONE. Now double-click 2_CHECK.bat again.
pause
