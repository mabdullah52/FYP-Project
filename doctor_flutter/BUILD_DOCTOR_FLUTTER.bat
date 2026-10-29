@echo off
title LungScan doctor app (Flutter) - build
cd /d "%~dp0"
echo ============================================================
echo  Builds the Flutter doctor app:
echo   1) web version  - opened from the AI server: http://127.0.0.1:8000/doctor/
echo   2) Android APK  - for a doctor's tablet on the same Wi-Fi (optional)
echo  Needs once: Flutter SDK (and Android Studio for the APK).
echo ============================================================
set FL=call "%~dp0..\flutterx.bat"
where flutter.bat >nul 2>nul
if errorlevel 1 (
  echo Flutter is not installed or not in PATH. See README_FLUTTER.txt
  pause
  exit /b
)
if not exist web (
  echo Creating the platform files once...
  %FL% create --org pk.edu.bahria --project-name lungscan_doctor --platforms web,android,windows .
  if exist test\widget_test.dart del test\widget_test.dart
)
if exist android\app\src\main\AndroidManifest.xml (
  powershell -NoProfile -Command ^
    "$f='android\app\src\main\AndroidManifest.xml'; $x=Get-Content $f -Raw;" ^
    "if($x -notmatch 'android.permission.INTERNET'){ $x=$x -replace '<application','<uses-permission android:name=\"android.permission.INTERNET\"/> <application' };" ^
    "if($x -notmatch 'usesCleartextTraffic'){ $x=$x -replace '<application','<application android:usesCleartextTraffic=\"true\"' };" ^
    "$x=$x -replace 'android:label=\"lungscan_doctor\"','android:label=\"LungScan Doctor\"';" ^
    "Set-Content $f $x -Encoding UTF8"
)
%FL% pub get
if errorlevel 1 goto failed
echo.
echo Building the web version...
%FL% build web --release --base-href /doctor/
if errorlevel 1 goto failed
echo.
echo  WEB VERSION READY. Start 7_DOCTOR_APP.bat - it now opens the Flutter app at
echo      http://127.0.0.1:8000/doctor/
echo.
choice /c YN /m "Also build the Android APK (needs Android Studio)"
if errorlevel 2 goto done
rem Windows fixes: use the NDK that is installed, and switch off Kotlin's incremental cache
rem (it fails when the project is on D: and the Flutter packages are on C:)
if exist android\gradle.properties (
  findstr /c:"kotlin.incremental=false" android\gradle.properties >nul || (echo.>> android\gradle.properties & echo kotlin.incremental=false>> android\gradle.properties)
)
set NDKV=
for /d %%n in ("%LOCALAPPDATA%\Android\Sdk\ndk\*") do set NDKV=%%~nxn
if defined NDKV if exist android\app\build.gradle.kts powershell -NoProfile -Command "$f='android\app\build.gradle.kts'; $x=Get-Content $f -Raw; $x=$x -replace 'ndkVersion = [^\r\n]+','ndkVersion = \"%NDKV%\"'; Set-Content $f $x -Encoding UTF8"
%FL% build apk --release
if errorlevel 1 goto failed
echo APK: build\app\outputs\flutter-apk\app-release.apk  (copy to the tablet and install)
:done
pause
exit /b
:failed
echo.
echo BUILD FAILED - copy the red error lines above (or a screenshot) and send them.
pause
