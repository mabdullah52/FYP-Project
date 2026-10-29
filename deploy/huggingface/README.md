---
title: LungScan AI
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# LungScan AI - online demo (FYP, Bahria University Lahore)

Lung nodule detection (MSGF-Net, LUNA16) with size, calibrated probability, uncertainty, Grad-CAM and
LUNA25-trained cancer-risk estimate, plus a patient portal.

* Doctor workstation: `/` (classic) and `/doctor/` (Flutter web app)
* Patient web app: `/patient/`  - the LungCare Android app uses `https://<this-space>.hf.space/patient` as server address

Runs on CPU, so one scan takes several minutes. Research prototype: decision support only, not a medical diagnosis.
Authors: Abdullah Asim, Hussnain Khalid. Supervisor: Rabia Masood.
