@echo off
title LungCare Android app - build the APK
cd /d "%~dp0"
echo ============================================================
echo  Builds the LungCare patient app for Android (.apk).
echo  Needs once: Flutter SDK + Android Studio (Android SDK, NDK) installed,
echo  and "flutterx doctor" showing a green tick for Android toolchain.
echo ============================================================
set FL=call "%~dp0..\flutterx.bat"
where flutter.bat >nul 2>nul
if errorlevel 1 (
  echo Flutter is not installed or not in PATH. See README_ANDROID.txt
  pause
  exit /b
)
if not exist android (
  echo Creating the Android project files once...
  %FL% create --org pk.edu.bahria --project-name lungcare_mobile --platforms android .
  if exist test\widget_test.dart del test\widget_test.dart
)
if not exist android\app\src\main\AndroidManifest.xml goto failed
echo Allowing the app to use the network and plain http on the hospital Wi-Fi...
powershell -NoProfile -Command ^
  "$f='android\app\src\main\AndroidManifest.xml'; $x=Get-Content $f -Raw;" ^
  "if($x -notmatch 'android.permission.INTERNET'){ $x=$x -replace '<application','<uses-permission android:name=\"android.permission.INTERNET\"/> <application' };" ^
  "if($x -notmatch 'usesCleartextTraffic'){ $x=$x -replace '<application','<application android:usesCleartextTraffic=\"true\"' };" ^
  "$x=$x -replace 'android:label=\"lungcare_mobile\"','android:label=\"LungCare\"';" ^
  "Set-Content $f $x -Encoding UTF8"
rem Windows fixes: use the NDK that is installed, and switch off Kotlin's incremental cache
rem (it fails when the project is on D: and the Flutter packages are on C:)
findstr /c:"kotlin.incremental=false" android\gradle.properties >nul || (echo.>> android\gradle.properties & echo kotlin.incremental=false>> android\gradle.properties)
rem Android Gradle Plugin 9 switches on a new DSL that the Flutter Gradle plugin does not support yet
findstr /r /c:"com.android.application.*version.*\"9\." android\settings.gradle.kts >nul 2>nul && (findstr /c:"android.newDsl" android\gradle.properties >nul || (echo android.newDsl=false>> android\gradle.properties))
set NDKV=
for /d %%n in ("%LOCALAPPDATA%\Android\Sdk\ndk\*") do set NDKV=%%~nxn
if defined NDKV powershell -NoProfile -Command "$f='android\app\build.gradle.kts'; $x=Get-Content $f -Raw; $x=$x -replace 'ndkVersion = [^\r\n]+','ndkVersion = \"%NDKV%\"'; Set-Content $f $x -Encoding UTF8"
%FL% pub get
if errorlevel 1 goto failed
%FL% build apk --release
if errorlevel 1 goto failed
echo.
echo The app is here:  build\app\outputs\flutter-apk\app-release.apk
echo Copy it to the phone and open it to install (allow "install unknown apps").
pause
exit /b
:failed
echo.
echo BUILD FAILED - copy the red error lines above (or a screenshot) and send them.
pause
