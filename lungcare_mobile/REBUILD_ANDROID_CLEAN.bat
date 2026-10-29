@echo off
title LungCare Android app - clean rebuild
cd /d "%~dp0"
echo ============================================================
echo  Clean rebuild of the LungCare Android app.
echo  The old Android project files are moved to android_old and
echo  made again with the Flutter SDK you have now, then the APK is built.
echo  Your app code (lib folder) is NOT touched.
echo ============================================================
if exist android\gradlew.bat (
  pushd android
  call gradlew.bat --stop >nul 2>nul
  popd
)
if exist android_old rmdir /s /q android_old
if exist android ren android android_old
if exist build rmdir /s /q build
if exist .dart_tool rmdir /s /q .dart_tool
call "%~dp0SETUP_AND_BUILD_ANDROID.bat"
