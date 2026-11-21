@echo off
title STEP 39 - Pack all trained models for a GitHub Release
cd /d "%~dp0"
echo Collecting every work\runs\*\best.pt and config.json into models_all.zip ...
if exist models_all.zip del models_all.zip
if exist _models_pack rmdir /s /q _models_pack
for /d %%R in (work\runs\*) do (
  if exist "%%R\best.pt" (
    mkdir "_models_pack\%%~nxR" 2>nul
    copy /y "%%R\best.pt" "_models_pack\%%~nxR\" >nul
    if exist "%%R\config.json" copy /y "%%R\config.json" "_models_pack\%%~nxR\" >nul
    echo   %%~nxR
  )
)
powershell -NoProfile -Command "Compress-Archive -Path _models_pack\* -DestinationPath models_all.zip -Force"
rmdir /s /q _models_pack
echo.
echo Done: models_all.zip
echo Upload it on GitHub: Releases - Draft a new release - attach models_all.zip - Publish.
pause
