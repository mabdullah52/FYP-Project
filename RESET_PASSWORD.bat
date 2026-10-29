@echo off
title LungScan AI - reset a forgotten password
cd /d "%~dp0"
lungenv\Scripts\python.exe doctor_app\reset_password.py
echo.
pause
