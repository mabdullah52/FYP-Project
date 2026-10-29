"""LungScan AI - clinician workstation (doctor web app).

    python doctor_app/server.py            -> open http://127.0.0.1:8000

* patients + CT studies stored locally (SQLite in app_data/)
* upload .mhd+.raw, .mha (LUNA25), a .zip (mhd/raw or DICOM), a DICOM folder's files, .nii/.nrrd,
  or give a path on this PC (fast for big LUNA16 scans)
* background analysis with the trained MSGF-Net (one scan at a time)
* slice viewer, per-finding review (confirm / reject / unsure), impression,
  printable report, and "release to patient" (feeds the patient app via access code)
Decision support for a radiologist - not a diagnostic device.
"""
import io
import json
import os
import queue
import secrets
import shutil
import sqlite3
import sys
import threading
import time
import traceback
import zipfile
from collections import OrderedDict
from contextlib import contextmanager
from typing import List, Optional
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import config as C  # noqa: E402
from brock import band as brock_band, brock_risk  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auth  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("LUNGSCAN_APP_DATA", os.path.join(ROOT, "app_data"))
STUDY_DIR = os.path.join(DATA_DIR, "studies")
DB_PATH = os.path.join(DATA_DIR, "lungscan.db")
os.makedirs(STUDY_DIR, exist_ok=True)


def find_model():
    """Newest-best checkpoint: env var, else round-2 model, else round-1, else any run."""
    env = os.environ.get("LUNGSCAN_MODEL")
    if env:
        return env
    for name in (getattr(C, "FINAL_RUN", "msgf_f9"), "msgf_f9", "msgf_f9_r2"):
        p = os.path.join(C.RUNS_DIR, name, "best.pt")
        if os.path.exists(p):
            return p
    cands = []
    if os.path.isdir(C.RUNS_DIR):
        for r in os.listdir(C.RUNS_DIR):
            p = os.path.join(C.RUNS_DIR, r, "best.pt")
            if os.path.exists(p):
                cands.append((os.path.getmtime(p), p))
    return max(cands)[1] if cands else None


# ------------------------------------------------------------------------------ database
_db_lock = threading.Lock()


@contextmanager
def tx():
    """One locked transaction: commit on success, always close the connection."""
    with _db_lock:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()


def init_db():
    with tx() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS patients(
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, mrn TEXT, age INTEGER,
            sex TEXT, phone TEXT, notes TEXT, access_code TEXT UNIQUE, created TEXT);
        CREATE TABLE IF NOT EXISTS studies(
            id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INTEGER NOT NULL, created TEXT,
            status TEXT, source TEXT, scan_path TEXT, report TEXT, error TEXT,
            impression TEXT, reviewed_by TEXT, reviewed_at TEXT, released INTEGER DEFAULT 0,
            released_at TEXT);
        CREATE TABLE IF NOT EXISTS patient_requests(
            id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INTEGER NOT NULL, kind TEXT, text TEXT,
            urgency TEXT, created TEXT, status TEXT DEFAULT 'open', reply TEXT, replied_by TEXT, replied_at TEXT);
        CREATE TABLE IF NOT EXISTS reviews(
            study_id INTEGER, finding_id INTEGER, decision TEXT, note TEXT, doctor TEXT, at TEXT,
            PRIMARY KEY(study_id, finding_id));
        """)
        con.execute("""CREATE TABLE IF NOT EXISTS finding_attrs(
            study_id INTEGER, finding_id INTEGER, nodule_type TEXT, upper_lobe INTEGER, spiculated INTEGER,
            PRIMARY KEY(study_id, finding_id))""")
        cols = {r["name"] for r in con.execute("PRAGMA table_info(patients)")}
        for c, t in (("smoking_status", "TEXT"), ("pack_years", "REAL"), ("family_history", "INTEGER"),
                     ("emphysema", "INTEGER")):
            if c not in cols:
                con.execute(f"ALTER TABLE patients ADD COLUMN {c} {t}")
        auth.init(con)
        # studies interrupted by a server stop go back into the queue on restart
        con.execute("UPDATE studies SET status='queued' WHERE status='processing'")


def now():
    return datetime.now().isoformat(timespec="seconds")


def row(r):
    return dict(r) if r is not None else None


# ------------------------------------------------------------------------------ analysis worker
class Analyzer:
    def __init__(self):
        self.q = queue.Queue()
        self.cad = None
        self.model_path = find_model()
        self.state = "not loaded"
        self.error = None
        threading.Thread(target=self._run, daemon=True).start()

    def _load(self):
        if self.cad is None:
            if not self.model_path:
                raise RuntimeError("No trained model found in work/runs - train first (4_TRAIN.bat).")
            self.state = "loading"
            from infer import LungCAD
            self.cad = LungCAD(self.model_path, device=os.environ.get("LUNGSCAN_DEVICE"))
            self.state = f"ready ({self.cad.pred.device.type}" + (", with cancer-risk model)" if self.cad.risk else ")")

    def submit(self, study_id):
        self.q.put(study_id)

    def _run(self):
        try:
            self._load()
        except Exception as e:
            self.state, self.error = "error", str(e)
        with tx() as con:
            for r in con.execute("SELECT id FROM studies WHERE status='queued' ORDER BY id"):
                self.q.put(r["id"])
        while True:
            sid = self.q.get()
            with tx() as con:
                s = con.execute("SELECT * FROM studies WHERE id=?", (sid,)).fetchone()
                if s is None or s["status"] not in ("queued",):
                    continue
                con.execute("UPDATE studies SET status='processing' WHERE id=?", (sid,))
            try:
                self._load()
                out = os.path.join(STUDY_DIR, str(sid), "results")
                with tx() as con:
                    pr = con.execute("SELECT age FROM patients WHERE id=?", (s["patient_id"],)).fetchone()
                    st = auth.get_settings(con)
                age = pr["age"] if pr is not None and pr["age"] else None
                rep = self.cad.analyze(s["scan_path"], out_dir=out, age=age, threshold=float(st["detection_threshold"]),
                                       review_uncertainty=float(st["review_uncertainty"]),
                                       max_findings=int(st["max_findings"]),
                                       fast=bool(int(st.get("fast_mode", 1))))
                geo = scan_cache.geometry(sid, s["scan_path"])
                rep["geometry"] = geo
                rep["model"] = os.path.relpath(self.model_path, ROOT)
                with tx() as con:
                    con.execute("UPDATE studies SET status='done', report=?, error=NULL WHERE id=?",
                                (json.dumps(rep), sid))
            except Exception as e:
                traceback.print_exc()
                with tx() as con:
                    con.execute("UPDATE studies SET status='error', error=? WHERE id=?", (str(e)[:500], sid))


class ScanCache:
    """Keeps the last 2 opened scans in memory for fast slice scrolling."""

    def __init__(self, size=2):
        self.size, self.items, self.lock = size, OrderedDict(), threading.Lock()

    def get(self, sid, path):
        with self.lock:
            if sid in self.items:
                self.items.move_to_end(sid)
                return self.items[sid]
        from luna_io import Scan
        sc = Scan(path)
        with self.lock:
            self.items[sid] = sc
            while len(self.items) > self.size:
                self.items.popitem(last=False)
        return sc

    def geometry(self, sid, path):
        sc = self.get(sid, path)
        z, y, x = sc.array.shape
        return {"slices": int(z), "rows": int(y), "cols": int(x),
                "spacing_xyz_mm": [round(float(v), 4) for v in sc.spacing],
                "direction_sign_xyz": [1 if sc.direction[i, i] >= 0 else -1 for i in range(3)]}


scan_cache = ScanCache()
init_db()
analyzer = Analyzer()
app = FastAPI(title="LungScan AI - clinician workstation")

# ------------------------------------------------------------------------------ authentication
OPEN_API = ("/api/health", "/api/auth/", "/api/patient-portal/")   # patient-portal is protected by the patient code


def session_token(request):
    """Session token from the browser cookie, an 'Authorization: Bearer' header (Flutter app) or
    ?token= (report / image links opened from the Flutter app)."""
    h = request.headers.get("authorization", "")
    if h.lower().startswith("bearer "):
        return h[7:].strip()
    return request.cookies.get(auth.COOKIE) or request.query_params.get("token")


@app.middleware("http")
async def require_login(request: Request, call_next):
    path = request.url.path
    if request.method == "OPTIONS":
        return await call_next(request)
    if (path.startswith("/api/") or path.startswith("/report/")) and not path.startswith(OPEN_API):
        with tx() as con:
            u = auth.user_for(con, session_token(request))
        if u is None:
            return JSONResponse({"detail": "Please log in."}, status_code=401)
        if path.startswith("/api/admin/") and u["role"] != "admin":
            return JSONResponse({"detail": "Administrator only."}, status_code=403)
        request.state.user = u
    return await call_next(request)


# added after require_login so it is the OUTER layer: also 401 replies carry CORS headers
# (needed only when the Flutter web app is run from another port during development)
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def who(request):
    return getattr(request.state, "user", None) or {"username": "system", "name": "system", "role": "doctor"}


def log(request, action, detail=""):
    with tx() as con:
        auth.audit(con, who(request), action, detail)


# ------------------------------------------------------------------------------ helpers
def get_study(sid):
    with tx() as con:
        s = con.execute("SELECT * FROM studies WHERE id=?", (sid,)).fetchone()
    if s is None:
        raise HTTPException(404, "study not found")
    return dict(s)


def study_payload(s, with_reviews=True):
    s = dict(s)
    s["report"] = json.loads(s["report"]) if s.get("report") else None
    if with_reviews:
        with tx() as con:
            s["reviews"] = {r["finding_id"]: row(r) for r in
                            con.execute("SELECT * FROM reviews WHERE study_id=?", (s["id"],))}
    s.pop("scan_path", None)
    return s


def locate_scan(folder):
    """Find the CT volume inside an extracted upload: .mhd, NIfTI/NRRD, or a DICOM series."""
    import SimpleITK as sitk
    for pat in (".mhd", ".mha", ".nii.gz", ".nii", ".nrrd"):
        for dp, _, fs in os.walk(folder):
            for f in sorted(fs):
                if f.lower().endswith(pat):
                    p = os.path.join(dp, f)
                    if pat == ".mhd":
                        raw = os.path.splitext(p)[0] + ".raw"
                        zraw = os.path.splitext(p)[0] + ".zraw"
                        if not (os.path.exists(raw) or os.path.exists(zraw)):
                            raise HTTPException(400, f"{f} was uploaded without its .raw file - select BOTH files.")
                    return p
    best, best_n = None, 0
    for dp, _, _ in os.walk(folder):
        for sid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(dp) or []:
            n = len(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dp, sid))
            if n > best_n:
                best, best_n = dp, n
    if best and best_n >= 10:
        return best
    raise HTTPException(400, "No CT volume found. Upload .mhd + .raw, .mha, a .zip, NIfTI, or DICOM slices.")


def new_study(patient_id, source):
    with tx() as con:
        if con.execute("SELECT 1 FROM patients WHERE id=?", (patient_id,)).fetchone() is None:
            raise HTTPException(404, "patient not found")
        cur = con.execute("INSERT INTO studies(patient_id, created, status, source) VALUES(?,?,?,?)",
                          (patient_id, now(), "uploading", source))
        return cur.lastrowid


def queue_study(sid, scan_path):
    with tx() as con:
        con.execute("UPDATE studies SET scan_path=?, status='queued' WHERE id=?", (scan_path, sid))
    analyzer.submit(sid)


# ------------------------------------------------------------------------------ API
class PatientIn(BaseModel):
    name: str
    mrn: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    phone: Optional[str] = None
    notes: Optional[str] = None
    smoking_status: Optional[str] = None      # never | former | current
    pack_years: Optional[float] = None
    family_history: Optional[bool] = None     # lung cancer in a first-degree relative
    emphysema: Optional[bool] = None


class LoginIn(BaseModel):
    username: str
    password: str


class SetupIn(BaseModel):
    username: str
    name: str
    password: str


class UserIn(BaseModel):
    username: str
    name: str
    role: str = "doctor"
    password: str


class UserPatch(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    active: Optional[bool] = None
    password: Optional[str] = None


class AttrsIn(BaseModel):
    nodule_type: Optional[str] = "solid"     # solid | part-solid | nonsolid
    upper_lobe: Optional[bool] = False
    spiculated: Optional[bool] = False


class PathIn(BaseModel):
    patient_id: int
    path: str


class ReviewIn(BaseModel):
    decision: str            # confirm | reject | unsure
    note: Optional[str] = None
    doctor: Optional[str] = None


class ImpressionIn(BaseModel):
    impression: str
    doctor: Optional[str] = None


def _session_response(token, payload):
    r = JSONResponse(payload)
    r.set_cookie(auth.COOKIE, token, httponly=True, samesite="lax", max_age=auth.SESSION_HOURS * 3600)
    return r


@app.get("/api/auth/state")
def auth_state(request: Request):
    with tx() as con:
        return {"setup_needed": auth.n_users(con) == 0, "user": auth.user_for(con, session_token(request))}


@app.post("/api/auth/setup")
def auth_setup(b: SetupIn):
    if len(b.password) < 8 or not b.username.strip():
        raise HTTPException(400, "Choose a username and a password of at least 8 characters.")
    with tx() as con:
        if auth.n_users(con) > 0:
            raise HTTPException(409, "The administrator account already exists.")
        auth.create_user(con, b.username, b.name, "admin", b.password)
        auth.audit(con, b.username.lower(), "setup", "administrator account created")
        token = auth.login(con, b.username, b.password)
        user = auth.user_for(con, token)
    return _session_response(token, {"user": user, "token": token})


@app.post("/api/auth/login")
def auth_login(b: LoginIn):
    with tx() as con:
        token = auth.login(con, b.username, b.password)
        auth.audit(con, b.username.lower(), "login" if token else "login failed")
        if not token:
            raise HTTPException(401, "Wrong username or password.")
        user = auth.user_for(con, token)
    return _session_response(token, {"user": user, "token": token})


@app.post("/api/auth/logout")
def auth_logout(request: Request):
    with tx() as con:
        auth.logout(con, session_token(request))
    r = JSONResponse({"ok": True})
    r.delete_cookie(auth.COOKIE)
    return r


# ---- administrator ---------------------------------------------------------------------
@app.get("/api/admin/users")
def admin_users():
    with tx() as con:
        return [row(r) for r in con.execute("SELECT id, username, name, role, active, created, last_login FROM users")]


@app.post("/api/admin/users")
def admin_add_user(b: UserIn, request: Request):
    if b.role not in ("doctor", "admin") or len(b.password) < 8:
        raise HTTPException(400, "Role must be doctor or admin; password at least 8 characters.")
    with tx() as con:
        try:
            auth.create_user(con, b.username, b.name, b.role, b.password)
        except sqlite3.IntegrityError:
            raise HTTPException(409, "That username already exists.")
        auth.audit(con, who(request), "user added", f"{b.username} ({b.role})")
    return {"ok": True}


@app.put("/api/admin/users/{uid}")
def admin_patch_user(uid: int, b: UserPatch, request: Request):
    with tx() as con:
        u = con.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        if u is None:
            raise HTTPException(404, "user not found")
        if uid == who(request)["id"] and (b.active is False or (b.role and b.role != "admin")):
            raise HTTPException(400, "You cannot deactivate or demote your own account.")
        if b.name is not None:
            con.execute("UPDATE users SET name=? WHERE id=?", (b.name, uid))
        if b.role in ("doctor", "admin"):
            con.execute("UPDATE users SET role=? WHERE id=?", (b.role, uid))
        if b.active is not None:
            con.execute("UPDATE users SET active=? WHERE id=?", (int(b.active), uid))
            if not b.active:
                con.execute("DELETE FROM sessions WHERE user_id=?", (uid,))
        if b.password:
            if len(b.password) < 8:
                raise HTTPException(400, "Password at least 8 characters.")
            con.execute("UPDATE users SET pw_hash=? WHERE id=?", (auth.hash_password(b.password), uid))
        auth.audit(con, who(request), "user changed", f"{u['username']}: {b.model_dump(exclude_none=True, exclude={'password'})}"
                   + (" + password reset" if b.password else ""))
    return {"ok": True}


@app.get("/api/settings")
def get_settings_api():
    with tx() as con:
        return auth.get_settings(con)


@app.put("/api/admin/settings")
def admin_settings(values: dict, request: Request):
    clean = {}
    for k, v in values.items():
        if k not in auth.DEFAULT_SETTINGS:
            continue
        v = float(v)
        if k == "max_findings":
            v = int(min(30, max(1, v)))
        elif k == "fast_mode":
            v = 1 if v else 0
        elif not 0.0 < v < 1.0:
            raise HTTPException(400, f"{k} must be between 0 and 1")
        clean[k] = v
    if clean.get("slice_yellow", 0.3) >= clean.get("slice_red", 0.7):
        raise HTTPException(400, "The yellow threshold must be below the red threshold.")
    with tx() as con:
        auth.set_settings(con, clean)
        auth.audit(con, who(request), "AI settings changed", json.dumps(clean))
        return auth.get_settings(con)


@app.get("/api/admin/audit")
def admin_audit(limit: int = 300):
    with tx() as con:
        return [row(r) for r in con.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (min(limit, 2000),))]


@app.delete("/api/admin/patients/{pid}")
def admin_delete_patient(pid: int, request: Request):
    with tx() as con:
        p = con.execute("SELECT name FROM patients WHERE id=?", (pid,)).fetchone()
        if p is None:
            raise HTTPException(404, "patient not found")
        sids = [r["id"] for r in con.execute("SELECT id FROM studies WHERE patient_id=?", (pid,))]
        for sid in sids:
            con.execute("DELETE FROM reviews WHERE study_id=?", (sid,))
            con.execute("DELETE FROM finding_attrs WHERE study_id=?", (sid,))
        con.execute("DELETE FROM studies WHERE patient_id=?", (pid,))
        con.execute("DELETE FROM patient_requests WHERE patient_id=?", (pid,))
        con.execute("DELETE FROM patients WHERE id=?", (pid,))
        auth.audit(con, who(request), "patient deleted", f"#{pid} {p['name']} ({len(sids)} studies)")
    for sid in sids:
        shutil.rmtree(os.path.join(STUDY_DIR, str(sid)), ignore_errors=True)
    return {"ok": True}


@app.get("/api/me")
def me(request: Request):
    return who(request)


@app.get("/api/health")
def health():
    return {"model": os.path.relpath(analyzer.model_path, ROOT) if analyzer.model_path else None,
            "state": analyzer.state, "error": analyzer.error, "queue": analyzer.q.qsize()}


@app.get("/api/patients")
def list_patients(q: str = ""):
    with tx() as con:
        rows = con.execute("""
            SELECT p.*, COUNT(s.id) AS n_studies, MAX(s.created) AS last_study,
                   (SELECT COUNT(*) FROM patient_requests r WHERE r.patient_id=p.id AND r.status='open') AS n_open
            FROM patients p LEFT JOIN studies s ON s.patient_id = p.id
            WHERE p.name LIKE ? OR IFNULL(p.mrn,'') LIKE ?
            GROUP BY p.id ORDER BY COALESCE(MAX(s.created), p.created) DESC""", (f"%{q}%", f"%{q}%")).fetchall()
    return [row(r) for r in rows]


PATIENT_FIELDS = ("name", "mrn", "age", "sex", "phone", "notes", "smoking_status", "pack_years", "family_history", "emphysema")


def _patient_values(p):
    d = p.model_dump()
    for k in ("family_history", "emphysema"):
        d[k] = None if d[k] is None else int(d[k])
    if d["smoking_status"] not in (None, "never", "former", "current"):
        raise HTTPException(400, "smoking status must be never, former or current")
    return [d[k].strip() if k == "name" else d[k] for k in PATIENT_FIELDS]


@app.post("/api/patients")
def create_patient(p: PatientIn, request: Request):
    if not p.name.strip():
        raise HTTPException(400, "name is required")
    code = secrets.token_hex(3).upper()
    with tx() as con:
        cur = con.execute(f"INSERT INTO patients({', '.join(PATIENT_FIELDS)}, access_code, created) "
                          f"VALUES({', '.join('?' * len(PATIENT_FIELDS))},?,?)", (*_patient_values(p), code, now()))
        auth.audit(con, who(request), "patient created", f"#{cur.lastrowid} {p.name.strip()}")
        return row(con.execute("SELECT * FROM patients WHERE id=?", (cur.lastrowid,)).fetchone())


@app.put("/api/patients/{pid}")
def update_patient(pid: int, p: PatientIn, request: Request):
    if not p.name.strip():
        raise HTTPException(400, "name is required")
    with tx() as con:
        if con.execute("SELECT 1 FROM patients WHERE id=?", (pid,)).fetchone() is None:
            raise HTTPException(404, "patient not found")
        con.execute(f"UPDATE patients SET {', '.join(k + '=?' for k in PATIENT_FIELDS)} WHERE id=?", (*_patient_values(p), pid))
        auth.audit(con, who(request), "patient edited", f"#{pid}")
        return row(con.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone())


@app.get("/api/patients/{pid}")
def get_patient(pid: int):
    with tx() as con:
        p = con.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
        if p is None:
            raise HTTPException(404, "patient not found")
        studies = con.execute("SELECT * FROM studies WHERE patient_id=? ORDER BY id DESC", (pid,)).fetchall()
    out = row(p)
    out["studies"] = [study_payload(s, with_reviews=False) for s in studies]
    return out


@app.post("/api/studies/upload")
async def upload_study(request: Request, patient_id: int = Form(...), files: List[UploadFile] = File(...)):
    sid = new_study(patient_id, ", ".join(f.filename for f in files)[:200])
    folder = os.path.join(STUDY_DIR, str(sid), "input")
    os.makedirs(folder, exist_ok=True)
    try:
        for f in files:
            name = os.path.basename(f.filename or "file")
            dest = os.path.join(folder, name)
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out, length=16 * 1024 * 1024)
            if name.lower().endswith(".zip"):
                with zipfile.ZipFile(dest) as z:
                    z.extractall(os.path.join(folder, os.path.splitext(name)[0]))
                os.remove(dest)
        queue_study(sid, locate_scan(folder))
        log(request, "CT uploaded", f"patient {patient_id}, study {sid}")
    except (HTTPException, zipfile.BadZipFile) as e:
        # nothing usable was uploaded: remove the empty study again and tell the doctor why
        with tx() as con:
            con.execute("DELETE FROM studies WHERE id=?", (sid,))
        shutil.rmtree(os.path.join(STUDY_DIR, str(sid)), ignore_errors=True)
        raise e if isinstance(e, HTTPException) else HTTPException(400, "The .zip file is damaged.")
    return {"id": sid}


@app.post("/api/studies/from_path")
def study_from_path(p: PathIn, request: Request):
    path = p.path.strip().strip('"')
    if not os.path.exists(path):
        raise HTTPException(400, f"Path not found on this PC: {path}")
    if os.path.isfile(path) and path.lower().endswith(".mhd") and not os.path.exists(os.path.splitext(path)[0] + ".raw"):
        raise HTTPException(400, "The .raw file must be next to the .mhd file.")
    sid = new_study(p.patient_id, path[:200])
    queue_study(sid, path if os.path.isfile(path) else locate_scan(path))
    log(request, "CT added from path", f"patient {p.patient_id}, study {sid}")
    return {"id": sid}


def add_brock(out, con):
    """Clinical (Brock/PanCan) malignancy risk per finding from the patient's data + nodule features."""
    rep, p = out.get("report"), out["patient"]
    if not rep:
        return
    attrs = {r["finding_id"]: row(r) for r in con.execute("SELECT * FROM finding_attrs WHERE study_id=?", (out["id"],))}
    rejected = {fid for fid, r in (out.get("reviews") or {}).items() if (r or {}).get("decision") == "reject"}
    count = max(1, sum(1 for f in rep.get("findings", []) if f.get("above_threshold") and f["id"] not in rejected))
    for f in rep.get("findings", []):
        a = attrs.get(f["id"]) or {"nodule_type": "solid", "upper_lobe": 0, "spiculated": 0}
        r = brock_risk(p.get("age"), (p.get("sex") or "").lower().startswith("f"), p.get("family_history"),
                       p.get("emphysema"), f["diameter_mm"], a["nodule_type"], a["upper_lobe"], count, a["spiculated"])
        f["attrs"] = {"nodule_type": a["nodule_type"], "upper_lobe": bool(a["upper_lobe"]), "spiculated": bool(a["spiculated"])}
        f["brock_risk"] = None if r is None else round(r, 4)
        f["brock_band"] = brock_band(r)


@app.get("/api/studies/{sid}")
def get_study_api(sid: int):
    s = get_study(sid)
    out = study_payload(s)
    with tx() as con:
        out["patient"] = row(con.execute("SELECT * FROM patients WHERE id=?", (s["patient_id"],)).fetchone())
        ahead = con.execute("SELECT COUNT(*) FROM studies WHERE status IN ('queued','processing') AND id<?",
                            (sid,)).fetchone()[0]
        add_brock(out, con)
        out["settings"] = auth.get_settings(con)
    out["queue_position"] = ahead
    return out


@app.put("/api/studies/{sid}/findings/{fid}/attrs")
def finding_attrs(sid: int, fid: int, b: AttrsIn, request: Request):
    if b.nodule_type not in ("solid", "part-solid", "nonsolid"):
        raise HTTPException(400, "nodule type must be solid, part-solid or nonsolid")
    get_study(sid)
    with tx() as con:
        con.execute("INSERT OR REPLACE INTO finding_attrs VALUES(?,?,?,?,?)",
                    (sid, fid, b.nodule_type, int(bool(b.upper_lobe)), int(bool(b.spiculated))))
        auth.audit(con, who(request), "nodule features", f"study {sid} finding {fid}: {b.model_dump()}")
    return get_study_api(sid)


@app.get("/api/studies/{sid}/cam/{fid}")
def finding_cam(sid: int, fid: int, z: int):
    """Grad-CAM overlay (RGBA PNG, 1 px = 1 mm, 32 x 32 mm around the finding) for CT slice z."""
    from PIL import Image
    from explain import cam_to_rgba
    s = study_payload(get_study(sid), with_reviews=False)
    f = next((x for x in (s.get("report") or {}).get("findings", []) if x["id"] == fid), None)
    if f is None or not f.get("cam"):
        raise HTTPException(404, "no heatmap for this finding (re-analyze the study)")
    cam = np.load(os.path.join(STUDY_DIR, str(sid), "results", os.path.basename(f["cam"]))).astype(np.float32) / 255
    geo = s["report"]["geometry"]
    sz = geo["spacing_xyz_mm"][2] * geo.get("direction_sign_xyz", [1, 1, 1])[2]
    k = int(round(cam.shape[0] / 2 - 0.5 + (z - f["voxel_zyx"][0]) * sz))
    if not 0 <= k < cam.shape[0]:
        return Response(status_code=204)
    sl = cam[k]
    sx, sy = geo.get("direction_sign_xyz", [1, 1, 1])[:2]
    sl = sl[::sy, ::sx]                                  # patient axes -> image axes
    buf = io.BytesIO()
    Image.fromarray(cam_to_rgba(sl), "RGBA").save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=600"})


@app.delete("/api/studies/{sid}")
def delete_study(sid: int, request: Request):
    get_study(sid)
    log(request, "study deleted", f"study {sid}")
    with tx() as con:
        con.execute("DELETE FROM reviews WHERE study_id=?", (sid,))
        con.execute("DELETE FROM studies WHERE id=?", (sid,))
    shutil.rmtree(os.path.join(STUDY_DIR, str(sid)), ignore_errors=True)
    return {"ok": True}


@app.get("/api/studies/{sid}/slice")
def get_slice(sid: int, z: int, window: str = "lung"):
    s = get_study(sid)
    if not s.get("scan_path"):
        raise HTTPException(409, "scan not available yet")
    from PIL import Image
    sc = scan_cache.get(sid, s["scan_path"])
    z = int(np.clip(z, 0, sc.array.shape[0] - 1))
    lo, hi = {"lung": (-1350, 150), "soft": (-160, 240), "bone": (-450, 1050)}.get(window, (-1350, 150))
    img = np.clip((sc.array[z].astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=False, compress_level=1)
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/studies/{sid}/files/{name}")
def study_file(sid: int, name: str):
    p = os.path.join(STUDY_DIR, str(sid), "results", os.path.basename(name))
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p)


@app.put("/api/studies/{sid}/findings/{fid}/review")
def review_finding(sid: int, fid: int, r: ReviewIn, request: Request):
    if r.decision not in ("confirm", "reject", "unsure"):
        raise HTTPException(400, "decision must be confirm, reject or unsure")
    get_study(sid)
    with tx() as con:
        con.execute("INSERT OR REPLACE INTO reviews VALUES(?,?,?,?,?,?)",
                    (sid, fid, r.decision, r.note, who(request)["name"], now()))
        auth.audit(con, who(request), "finding reviewed", f"study {sid} finding {fid}: {r.decision}")
    return {"ok": True}


@app.put("/api/studies/{sid}/impression")
def save_impression(sid: int, body: ImpressionIn, request: Request):
    get_study(sid)
    with tx() as con:
        con.execute("UPDATE studies SET impression=?, reviewed_by=?, reviewed_at=? WHERE id=?",
                    (body.impression, who(request)["name"], now(), sid))
        auth.audit(con, who(request), "impression saved", f"study {sid}")
    return {"ok": True}


@app.post("/api/studies/{sid}/release")
def release(sid: int, request: Request, released: bool = True):
    s = get_study(sid)
    if released and (s["status"] != "done" or not (s.get("impression") or "").strip()):
        raise HTTPException(400, "Write and save the doctor's impression before releasing to the patient.")
    with tx() as con:
        con.execute("UPDATE studies SET released=?, released_at=? WHERE id=?",
                    (1 if released else 0, now() if released else None, sid))
        auth.audit(con, who(request), "released" if released else "release withdrawn", f"study {sid}")
    return {"ok": True}


@app.post("/api/studies/{sid}/reanalyze")
def reanalyze(sid: int, request: Request):
    s = get_study(sid)
    if not s.get("scan_path"):
        raise HTTPException(400, "no scan stored for this study")
    queue_study(sid, s["scan_path"])
    log(request, "re-analyze", f"study {sid}")
    return {"ok": True}


# ---- patient-facing endpoint (used by the patient app): only RELEASED studies -----------
@app.get("/api/patient-portal/{code}")
def patient_portal(code: str):
    with tx() as con:
        p = con.execute("SELECT id, name, age, sex FROM patients WHERE access_code=?", (code.upper(),)).fetchone()
        if p is None:
            raise HTTPException(404, "unknown access code")
        ss = con.execute("SELECT * FROM studies WHERE patient_id=? AND released=1 ORDER BY id DESC",
                         (p["id"],)).fetchall()
    out = []
    for s in ss:
        d = study_payload(s)
        rep = d.get("report") or {}
        confirmed = [f for f in rep.get("findings", [])
                     if (d["reviews"].get(f["id"]) or {}).get("decision") != "reject"]
        out.append({"study_id": d["id"], "date": d["created"], "released_at": d["released_at"],
                    "doctor": d["reviewed_by"], "impression": d["impression"],
                    "findings": [{"id": f["id"], "diameter_mm": f["diameter_mm"],
                                  "lung_rads_size_category": f["lung_rads_size_category"],
                                  "doctor_decision": (d["reviews"].get(f["id"]) or {}).get("decision"),
                                  "doctor_note": (d["reviews"].get(f["id"]) or {}).get("note")}
                                 for f in confirmed]})
    with tx() as con:
        reqs = [row(r) for r in con.execute(
            "SELECT id, kind, text, urgency, created, status, reply, replied_by, replied_at FROM patient_requests "
            "WHERE patient_id=? ORDER BY id DESC LIMIT 50", (p["id"],))]
    return {"patient": row(p), "studies": out, "requests": reqs}


class RecoverIn(BaseModel):
    phone: str
    mrn: str


_RECOVER_TRIES = {}


def _digits(x):
    return "".join(ch for ch in str(x or "") if ch.isdigit())


@app.post("/api/patient-portal/recover")
def recover_code(b: RecoverIn, request: Request):
    """'Forgot your code?' in the patient app: phone number + MRN (both recorded by the clinic) must
    match the same patient. At most 5 attempts per 15 minutes per device; every attempt is audited."""
    ip = request.client.host if request.client else "?"
    t = datetime.now().timestamp()
    tries = [x for x in _RECOVER_TRIES.get(ip, []) if t - x < 900]
    if len(tries) >= 5:
        raise HTTPException(429, "Too many attempts. Please wait 15 minutes or ask the clinic for your code.")
    _RECOVER_TRIES[ip] = tries + [t]
    phone, mrn = _digits(b.phone), (b.mrn or "").strip().upper()
    if len(phone) < 7 or not mrn:
        raise HTTPException(400, "Enter the phone number and the MRN (hospital number) the clinic has on record.")
    with tx() as con:
        hit = None
        for r in con.execute("SELECT id, name, phone, mrn, access_code FROM patients WHERE UPPER(TRIM(IFNULL(mrn,'')))=?", (mrn,)):
            p = _digits(r["phone"])
            if p and (p[-10:] == phone[-10:]):
                hit = r
                break
        auth.audit(con, "patient-app", "code recovery " + ("ok" if hit else "failed"),
                   f"patient #{hit['id']}" if hit else f"MRN {mrn[:20]}")
    if hit is None:
        raise HTTPException(404, "No patient found with this phone number and MRN. Please ask the clinic for your code.")
    return {"code": hit["access_code"], "name": hit["name"]}


@app.post("/api/patients/{pid}/new-code")
def new_access_code(pid: int, request: Request):
    """Give the patient a new access code (e.g. code lost or shared); the old code stops working."""
    with tx() as con:
        if con.execute("SELECT 1 FROM patients WHERE id=?", (pid,)).fetchone() is None:
            raise HTTPException(404, "patient not found")
        while True:
            code = secrets.token_hex(3).upper()
            if con.execute("SELECT 1 FROM patients WHERE access_code=?", (code,)).fetchone() is None:
                break
        con.execute("UPDATE patients SET access_code=? WHERE id=?", (code, pid))
        auth.audit(con, who(request), "new patient code", f"patient #{pid}")
    return {"access_code": code}


class PatientRequestIn(BaseModel):
    kind: str                 # appointment | question | symptom
    text: str
    urgency: Optional[str] = "routine"     # routine | soon | urgent


@app.post("/api/patient-portal/{code}/requests")
def patient_request(code: str, r: PatientRequestIn):
    if r.kind not in ("appointment", "question", "symptom") or not r.text.strip():
        raise HTTPException(400, "invalid request")
    with tx() as con:
        p = con.execute("SELECT id FROM patients WHERE access_code=?", (code.upper(),)).fetchone()
        if p is None:
            raise HTTPException(404, "unknown access code")
        cur = con.execute("INSERT INTO patient_requests(patient_id, kind, text, urgency, created) VALUES(?,?,?,?,?)",
                          (p["id"], r.kind, r.text.strip()[:2000], r.urgency or "routine", now()))
        return {"id": cur.lastrowid, "status": "open"}


@app.get("/api/patients/{pid}/requests")
def list_requests(pid: int):
    with tx() as con:
        return [row(r) for r in con.execute("SELECT * FROM patient_requests WHERE patient_id=? ORDER BY "
                                            "status='open' DESC, urgency='urgent' DESC, id DESC", (pid,))]


class ReplyIn(BaseModel):
    reply: str
    doctor: Optional[str] = None


@app.put("/api/requests/{rid}")
def reply_request(rid: int, body: ReplyIn, request: Request):
    with tx() as con:
        if con.execute("SELECT 1 FROM patient_requests WHERE id=?", (rid,)).fetchone() is None:
            raise HTTPException(404, "request not found")
        con.execute("UPDATE patient_requests SET reply=?, replied_by=?, replied_at=?, status='answered' WHERE id=?",
                    (body.reply, who(request)["name"], now(), rid))
        auth.audit(con, who(request), "replied to patient", f"request {rid}")
    return {"ok": True}


# ------------------------------------------------------------------------------ printable report
@app.get("/report/{sid}", response_class=HTMLResponse)
def printable_report(sid: int, token: Optional[str] = None):
    import html
    from urllib.parse import quote
    d = get_study_api(sid)
    tq = f"?token={quote(token)}" if token else ""       # report opened from the Flutter app (no cookie)
    p, rep = d["patient"], d.get("report") or {}
    sm = rep.get("summary", {})
    e = lambda x: html.escape("" if x is None else str(x))
    dec_label = {"confirm": "Confirmed", "reject": "Rejected (false positive)", "unsure": "Uncertain"}
    rows = ""
    for f in rep.get("findings", []):
        rv = d["reviews"].get(f["id"]) or {}
        img = f'<img src="/api/studies/{sid}/files/{e(f.get("snapshot"))}{tq}">' if f.get("snapshot") else ""
        if f.get("cam_snapshot"):
            img += f'<img src="/api/studies/{sid}/files/{e(f.get("cam_snapshot"))}{tq}">'
        risk = f"{f['cancer_risk'] * 100:.1f}% ({e(f['cancer_risk_band'])})" if f.get("cancer_risk") is not None else "-"
        brk = f"{f['brock_risk'] * 100:.1f}% ({e(f['brock_band'])})" if f.get("brock_risk") is not None else "-"
        rows += f"""<tr><td>{f['id']}</td><td>{f.get('slice_index', f['voxel_zyx'][0])}</td>
        <td>{f['diameter_mm']:.1f} mm</td><td>{f['probability'] * 100:.1f}%</td>
        <td>{'Yes' if f['needs_review'] else 'No'}</td><td>{e(f['lung_rads_size_category'])}</td>
        <td>AI: {risk}<br>Brock: {brk}</td>
        <td>{e(dec_label.get(rv.get('decision'), 'Not reviewed'))}{('<br><small>' + e(rv.get('note')) + '</small>') if rv.get('note') else ''}</td>
        <td class="img">{img}</td></tr>"""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Report {sid} - {e(p['name'])}</title>
<style>body{{font:13px/1.45 system-ui,Segoe UI,Arial;color:#111;max-width:900px;margin:24px auto;padding:0 16px}}
h1{{font-size:20px;margin:0}} .muted{{color:#555}} table{{border-collapse:collapse;width:100%;margin:12px 0}}
td,th{{border:1px solid #ccc;padding:6px;vertical-align:top;text-align:left}} th{{background:#f2f2f0}}
td.img img{{width:230px}} .box{{border:1px solid #ccc;padding:10px 12px;margin:10px 0;white-space:pre-wrap}}
.disc{{font-size:11px;color:#555;border-top:1px solid #ccc;margin-top:18px;padding-top:8px}}
@media print{{button{{display:none}}}}</style></head><body>
<button onclick="print()">Print / Save as PDF</button>
<h1>Chest CT - AI-assisted pulmonary nodule report</h1>
<p class="muted">Study #{sid} &middot; {e(d['created'])} &middot; model {e(rep.get('model'))}</p>
<table><tr><th>Patient</th><td>{e(p['name'])}</td><th>MRN</th><td>{e(p['mrn'])}</td></tr>
<tr><th>Age / Sex</th><td>{e(p['age'])} / {e(p['sex'])}</td><th>Reviewed by</th><td>{e(d['reviewed_by'])} {e(d['reviewed_at'])}</td></tr>
<tr><th>Smoking</th><td>{e(p.get('smoking_status') or '-')}{(', ' + e(p['pack_years']) + ' pack-years') if p.get('pack_years') else ''}</td>
<th>Family history / emphysema</th><td>{'yes' if p.get('family_history') else ('no' if p.get('family_history') == 0 else '-')} / {'yes' if p.get('emphysema') else ('no' if p.get('emphysema') == 0 else '-')}</td></tr></table>
<h3>AI summary</h3><div class="box">{e(sm.get('statement'))}
Candidates screened: {e(sm.get('n_candidates_screened'))} &middot; flagged for review: {e(sm.get('n_flagged_for_review'))}</div>
<h3>Findings</h3><table><tr><th>#</th><th>Slice</th><th>Size</th><th>AI probability</th><th>Needs review</th>
<th>Lung-RADS (size only)</th><th>Cancer risk (AI image / Brock clinical)</th><th>Doctor decision</th><th>Image</th></tr>{rows or '<tr><td colspan=9>No findings</td></tr>'}</table>
<h3>Radiologist impression</h3><div class="box">{e(d.get('impression')) or '<i>Not written yet</i>'}</div>
<p style="margin-top:40px">Signature: ______________________ &nbsp;&nbsp; Date: ______________</p>
<p class="disc">{e(sm.get('disclaimer'))} Probabilities come from MSGF-Net trained on LUNA16 and indicate nodule
presence, not malignancy. The AI cancer risk comes from a separate model trained on LUNA25 (NLST screening
data, test AUC about 0.9) and is a research estimate only. The Brock (PanCan) risk is the published clinical
model (McWilliams et al., NEJM 2013) computed from the patient's data and the nodule features recorded by the
doctor (solid nodule, non-upper-lobe and no spiculation are assumed unless changed). Heatmaps (Grad-CAM) show the
image regions that most influenced the AI's nodule decision. Lung-RADS categories shown are derived from estimated size only (solid nodule assumed).</p>
</body></html>"""


FLUTTER_WEB = os.path.join(ROOT, "doctor_flutter", "build", "web")
if os.path.isdir(FLUTTER_WEB):          # Flutter doctor app (built with BUILD_DOCTOR_FLUTTER.bat)
    app.mount("/doctor", StaticFiles(directory=FLUTTER_WEB, html=True), name="doctor_flutter")
app.mount("/", StaticFiles(directory=os.path.join(APP_DIR, "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("LUNGSCAN_PORT", "8000"))
    host = os.environ.get("LUNGSCAN_HOST", "127.0.0.1")
    print(f"\n  LungScan AI clinician workstation -> http://127.0.0.1:{port}"
          + ("   (Flutter doctor app: http://127.0.0.1:%d/doctor/)" % port if os.path.isdir(FLUTTER_WEB) else ""))
    if host == "0.0.0.0":
        import socket
        try:
            s_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s_.connect(("8.8.8.8", 80))
            print(f"  Android tablet / other PCs on the same network: http://{s_.getsockname()[0]}:{port}")
            s_.close()
        except OSError:
            pass
    print(f"  model: {analyzer.model_path}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
