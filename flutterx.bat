@echo off
rem Runs the Flutter tool directly with its own Dart, skipping flutter.bat's startup script
rem (which hangs on some university PCs). Use exactly like "flutter":  flutterx doctor -v
rem If several Flutter SDKs are in PATH, the first one whose folder is writable is used.
setlocal
set "FLBIN="
for /f "delims=" %%i in ('where flutter.bat 2^>nul') do call :try "%%~dpi"
if not defined FLBIN (
  echo No usable Flutter SDK found in PATH. See doctor_flutter\README_FLUTTER.txt
  exit /b 1
)
for %%a in ("%FLBIN%..") do set "FLUTTER_ROOT=%%~fa"
set "PATH=%FLBIN%;%PATH%"
set "PKG=%FLUTTER_ROOT%\packages\flutter_tools\.dart_tool\package_config.json"
set "PK="
if exist "%PKG%" set "PK=--packages=%PKG%"
"%FLBIN%cache\dart-sdk\bin\dart.exe" %PK% "%FLBIN%cache\flutter_tools.snapshot" %*
exit /b %errorlevel%

:try
if defined FLBIN exit /b 0
if not exist "%~1cache\flutter_tools.snapshot" exit /b 0
(type nul > "%~1cache\.flutterx_write_test") 2>nul || exit /b 0
del "%~1cache\.flutterx_write_test" 2>nul
set "FLBIN=%~1"
exit /b 0
