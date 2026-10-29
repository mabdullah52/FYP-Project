LungScan AI - doctor app in Flutter
====================================

How it works
  The AI model (PyTorch, GPU) runs in the Python server: 7_DOCTOR_APP.bat.
  This Flutter app is the doctor's screen. It talks to that server over the network:
    * web version  -> served by the server itself at  http://127.0.0.1:8000/doctor/
    * Android APK  -> on a tablet/phone on the same Wi-Fi; type the PC address shown in the
                      7_DOCTOR_APP window (e.g. http://192.168.1.20:8000) on the sign-in screen
    * Windows app  -> optional:  flutter run -d windows  (needs Visual Studio "Desktop C++")

Install Flutter once (on any Windows PC)
  1. Download the Flutter SDK zip from https://docs.flutter.dev/get-started/install/windows
     and unzip it to C:\flutter   (not inside "Program Files").
  2. Add C:\flutter\bin to the PATH (Start > "Edit the system environment variables" >
     Environment Variables > Path > New).
  3. Open a new Command Prompt and run:  flutter doctor
     - For the web version you only need Chrome or Edge.
     - For the APK also install Android Studio, then run:  flutter doctor --android-licenses

Build
  Double-click BUILD_DOCTOR_FLUTTER.bat (in this folder). The first build takes a few minutes.

Run
  Double-click 7_DOCTOR_APP.bat. The browser opens the Flutter app. Sign in with the same
  accounts as before (the old web page is still at http://127.0.0.1:8000/ as a fallback).

Features (same as the previous web app)
  sign-in with doctor/admin roles; patient list and search; new/edit patient with lifestyle data;
  upload CT (.mhd+.raw, .mha, NIfTI, zip, DICOM) or add from a path on the PC; AI analysis with
  live status; slice viewer (lung/soft tissue/bone windows, mouse wheel / arrow keys / touch drag,
  finding circles, slice risk bar green/yellow/red, Grad-CAM heatmap); findings with probability,
  size, uncertainty, Lung-RADS, AI cancer risk and Brock clinical risk; nodule type / upper lobe /
  spiculation; confirm / false positive / unsure with notes; impression; release to the patient app;
  printable report; re-analyze; patient messages with replies; admin: users, AI settings
  (thresholds, fast mode), activity log.

Forgot password
  Doctors: the administrator resets it in Administration > Users.
  Administrator: on the AI server PC double-click RESET_PASSWORD.bat (project folder).
