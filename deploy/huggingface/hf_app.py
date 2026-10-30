"""LungScan AI - one web server for the online demo (Hugging Face Spaces, port 7860).

    /            doctor workstation (classic web page)
    /doctor/     doctor app (Flutter web build)
    /api/...     doctor / AI API (also used by the Flutter doctor app)
    /patient/    LungCare patient web app; the Android app uses  https://<space>/patient  as server address

The administrator account is created from the Space secrets ADMIN_USER / ADMIN_PASSWORD, so nobody
else can claim it. Data lives in /data (persistent storage) if available, otherwise it resets on restart.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

def _writable_dir(*candidates):
    """First folder we can create and write in (the container user may not own /app)."""
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".write_test")
            open(probe, "w").close()
            os.remove(probe)
            return d
        except OSError:
            continue
    raise SystemExit("no writable folder for the app data")


DATA = _writable_dir("/data/app_data", os.path.join(ROOT, "app_data"), "/tmp/lungscan_app_data")
PORT = int(os.environ.get("PORT", "7860"))
os.environ.setdefault("LUNGSCAN_APP_DATA", DATA)
os.environ.setdefault("LUNGCARE_DATA", DATA)
os.environ.setdefault("LUNGSCAN_DEVICE", "cpu")


def _cpu_limit():
    """CPUs this container may really use (cgroup quota), not the host's core count."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, round(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


CPUS = _cpu_limit()
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):          # use every CPU Azure gives us, no more
    os.environ.setdefault(_v, str(CPUS))
# on a CPU the full 8-flip TTA + MC dropout is the slow part: give it to the 40 most suspicious
# candidates instead of 150 (findings shown are at most 10, so their scores are unchanged in practice)
os.environ.setdefault("LUNGSCAN_RESCORE_TOP", "40")
os.environ.setdefault("LUNGSCAN_FINDER_GPU", "0")
os.environ.setdefault("DOCTOR_API", f"http://127.0.0.1:{PORT}")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


doctor = load("doctor_server", "doctor_app/server.py")
patient = load("patient_server", "patient_app/server.py")

# administrator account from Space secrets (never stored in the code)
user, pw = os.environ.get("ADMIN_USER", "").strip(), os.environ.get("ADMIN_PASSWORD", "")
if user and len(pw) >= 8:
    import auth
    with doctor.tx() as con:
        if auth.n_users(con) == 0:
            auth.create_user(con, user, os.environ.get("ADMIN_NAME", user), "admin", pw)
            auth.audit(con, user.lower(), "setup", "administrator account created from Space secrets")
            print(f"administrator '{user}' created")
else:
    print("WARNING: set the Space secrets ADMIN_USER and ADMIN_PASSWORD (8+ characters) - "
          "until then the first visitor can create the administrator account.")

from starlette.applications import Starlette  # noqa: E402
from starlette.responses import RedirectResponse  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402

app = Starlette(routes=[
    Route("/patient", lambda r: RedirectResponse("/patient/")),
    Mount("/patient", app=patient.app),
    Mount("/", app=doctor.app),
])

if __name__ == "__main__":
    import uvicorn
    print(f"LungScan AI online on port {PORT}  (data: {DATA}, CPUs: {CPUS})")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
