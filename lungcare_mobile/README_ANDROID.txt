LungCare - Android patient app (Flutter)
========================================

What it does
  Results tab   : results released by the doctor, plain-language summary, next-scan date, reminders
  Assistant tab : the same AI assistant as the web patient app (explains results, sets reminders,
                  logs symptoms with red-flag escalation, messages the doctor, answers from the guideline library)
  Diary tab     : symptom diary (urgent symptoms alert the doctor)
  Doctor tab    : questions / appointment requests and the doctor's replies
  Guides tab    : the approved guideline library (WHO, Lung-RADS, USPSTF summaries + PDFs the hospital adds)

It talks to the patient server (9_PATIENT_APP.bat) over the hospital Wi-Fi.

1. Install once (on any Windows PC with internet)
   - Flutter SDK: https://docs.flutter.dev/get-started/install/windows/mobile
     (unzip e.g. to C:\flutter and add C:\flutter\bin to PATH)
   - Android Studio: https://developer.android.com/studio  (it installs the Android SDK)
   - In a new command window run:  flutter doctor --android-licenses   (answer y)
   - Then:  flutter doctor   -> "Android toolchain" must have a green tick

2. Build the app
   - Double-click SETUP_AND_BUILD_ANDROID.bat in this folder (first build takes 5-15 minutes)
   - Result: build\app\outputs\flutter-apk\app-release.apk

3. Install on the phone
   - Copy app-release.apk to the phone (USB / WhatsApp to yourself / Google Drive) and open it,
     allow "install unknown apps".

4. Use it
   - On the hospital PC: run 7_DOCTOR_APP.bat and 9_PATIENT_APP.bat. The patient-app window prints
     "Android app: connect to http://192.168.x.x:8100".
   - Phone on the SAME Wi-Fi. Open LungCare, type that address and the patient's access code
     (shown in the doctor app under the patient's name).
   - If the phone cannot connect: allow Python through the Windows Firewall when Windows asks
     (or: Windows Security > Firewall > Allow an app > python.exe > Private).

Run on a PC without a phone (for screenshots)
   flutter create --platforms windows .   then   flutter run -d windows   (needs Visual Studio "Desktop development with C++")
   or start an Android emulator in Android Studio and run:  flutter run
   On the emulator the PC is reachable as  http://10.0.2.2:8100
