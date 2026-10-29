"""LungCare - patient app (mobile-friendly web app with an agentic AI assistant).

    python patient_app/server.py        -> http://127.0.0.1:8100   (the doctor app must be running too)

* the patient logs in with the 6-character code shown in the doctor app
* sees ONLY results the doctor reviewed and released, explained in simple words
* agentic assistant (Claude with tools if ANTHROPIC_API_KEY is set, otherwise offline planner)
  that can set reminders, keep a symptom diary with red-flag triage, and send questions /
  appointment requests to the doctor (who answers in the doctor app)
"""
import json
import os
import sqlite3
import sys
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import date, datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Header, HTTPException  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from agent import OfflineBrain, Tools, _summary_text, make_brain  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
DOCTOR_API = os.environ.get("DOCTOR_API", "http://127.0.0.1:8000").rstrip("/")
DATA_DIR = os.environ.get("LUNGCARE_DATA", os.path.join(ROOT, "app_data"))
os.makedirs(DATA_DIR, exist_ok=True)
DB = os.path.join(DATA_DIR, "patient_app.db")
_lock = threading.Lock()


@contextmanager
def tx():
    with _lock:
        con = sqlite3.connect(DB)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()


with tx() as con:
    con.executescript("""
    CREATE TABLE IF NOT EXISTS chat(id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT, role TEXT, content TEXT,
                                    actions TEXT, at TEXT);
    CREATE TABLE IF NOT EXISTS reminders(id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT, title TEXT, due_date TEXT,
                                         note TEXT, done INTEGER DEFAULT 0, created TEXT);
    CREATE TABLE IF NOT EXISTS symptoms(id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT, symptom TEXT, severity INTEGER,
                                        notes TEXT, urgent INTEGER, at TEXT);
    """)


def now():
    return datetime.now().isoformat(timespec="seconds")


def doctor_api(path, payload=None):
    req = urllib.request.Request(DOCTOR_API + path, method="POST" if payload is not None else "GET",
                                 data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = json.loads(e.read() or b"{}").get("detail", str(e))
        raise HTTPException(e.code, detail)
    except urllib.error.URLError:
        raise HTTPException(503, "The hospital system is not reachable. Please try again later "
                                 "(is the doctor app running?).")


class PatientContext:
    """Everything the agent's tools may touch, scoped to ONE patient code."""

    def __init__(self, code):
        self.code = code.upper().strip()
        self._portal = None

    def portal(self):
        if self._portal is None:
            self._portal = doctor_api(f"/api/patient-portal/{self.code}")
        return self._portal

    def reminders(self):
        with tx() as con:
            return [dict(r) for r in con.execute("SELECT id, title, due_date, note, done FROM reminders WHERE code=? "
                                                 "ORDER BY done, due_date", (self.code,))]

    def add_reminder(self, title, due_date, note=None):
        with tx() as con:
            dup = con.execute("SELECT id FROM reminders WHERE code=? AND title=? AND due_date=?",
                              (self.code, title, due_date)).fetchone()
            if dup:
                return dup["id"]
            return con.execute("INSERT INTO reminders(code, title, due_date, note, created) VALUES(?,?,?,?,?)",
                               (self.code, title, due_date, note, now())).lastrowid

    def add_symptom(self, symptom, severity, notes, urgent):
        with tx() as con:
            con.execute("INSERT INTO symptoms(code, symptom, severity, notes, urgent, at) VALUES(?,?,?,?,?,?)",
                        (self.code, symptom, severity, notes, int(bool(urgent)), now()))

    def send_request(self, kind, text, urgency="routine"):
        self._portal = None
        return doctor_api(f"/api/patient-portal/{self.code}/requests", {"kind": kind, "text": text, "urgency": urgency})


brain = make_brain()
fallback = OfflineBrain()
app = FastAPI(title="LungCare patient app")


def ctx_from(code):
    if not code:
        raise HTTPException(401, "missing access code")
    c = PatientContext(code)
    c.portal()                       # validates the code (404 if unknown)
    return c


class LoginIn(BaseModel):
    code: str


class ChatIn(BaseModel):
    message: str


class ReminderIn(BaseModel):
    title: str
    due_date: str
    note: Optional[str] = None


class SymptomIn(BaseModel):
    symptom: str
    severity: Optional[int] = None
    notes: Optional[str] = None


class RequestIn(BaseModel):
    kind: str
    text: str
    urgency: Optional[str] = "routine"


@app.post("/api/login")
def login(b: LoginIn):
    c = ctx_from(b.code)
    return {"ok": True, "name": c.portal()["patient"]["name"], "code": c.code}


class RecoverIn(BaseModel):
    phone: str
    mrn: str


@app.post("/api/recover")
def recover(b: RecoverIn):
    """'Forgot your code?' - checked by the hospital system (phone + MRN must match one patient)."""
    return doctor_api("/api/patient-portal/recover", {"phone": b.phone, "mrn": b.mrn})


@app.get("/api/home")
def home(x_code: str = Header(None)):
    c = ctx_from(x_code)
    t = Tools(c)
    with tx() as con:
        symptoms = [dict(r) for r in con.execute("SELECT * FROM symptoms WHERE code=? ORDER BY id DESC LIMIT 30", (c.code,))]
    return {"patient": c.portal()["patient"], "studies": c.portal()["studies"], "requests": c.portal().get("requests", []),
            "summary": _summary_text(t.get_my_results()), "plan": t.get_followup_plan(),
            "reminders": c.reminders(), "symptoms": symptoms, "assistant": brain.name, "today": date.today().isoformat()}


@app.get("/api/chat")
def chat_history(x_code: str = Header(None)):
    c = ctx_from(x_code)
    with tx() as con:
        rows = [dict(r) for r in con.execute("SELECT role, content, actions, at FROM chat WHERE code=? ORDER BY id DESC "
                                             "LIMIT 60", (c.code,))][::-1]
    for r in rows:
        r["actions"] = json.loads(r["actions"] or "[]")
    return rows


@app.post("/api/chat")
def chat(b: ChatIn, x_code: str = Header(None)):
    c = ctx_from(x_code)
    msg = b.message.strip()[:2000]
    if not msg:
        raise HTTPException(400, "empty message")
    with tx() as con:
        hist = [{"role": r["role"], "content": r["content"]} for r in
                con.execute("SELECT role, content FROM chat WHERE code=? ORDER BY id DESC LIMIT 12", (c.code,))][::-1]
    tools = Tools(c)
    mode = brain.name
    try:
        reply = brain.run(tools, hist, msg)
    except Exception as e:            # network / API problem -> still answer, offline
        print("assistant error, falling back to offline:", e)
        tools = Tools(c)
        reply, mode = fallback.run(tools, hist, msg), "offline"
    with tx() as con:
        con.execute("INSERT INTO chat(code, role, content, actions, at) VALUES(?,?,?,?,?)", (c.code, "user", msg, "[]", now()))
        con.execute("INSERT INTO chat(code, role, content, actions, at) VALUES(?,?,?,?,?)",
                    (c.code, "assistant", reply, json.dumps(tools.actions), now()))
    return {"reply": reply, "actions": tools.actions, "mode": mode}


@app.post("/api/reminders")
def add_reminder(b: ReminderIn, x_code: str = Header(None)):
    c = ctx_from(x_code)
    r = Tools(c).set_reminder(b.title, b.due_date, b.note)
    if "error" in r:
        raise HTTPException(400, r["error"])
    return r


@app.put("/api/reminders/{rid}/done")
def reminder_done(rid: int, done: bool = True, x_code: str = Header(None)):
    c = ctx_from(x_code)
    with tx() as con:
        con.execute("UPDATE reminders SET done=? WHERE id=? AND code=?", (int(done), rid, c.code))
    return {"ok": True}


@app.delete("/api/reminders/{rid}")
def reminder_delete(rid: int, x_code: str = Header(None)):
    c = ctx_from(x_code)
    with tx() as con:
        con.execute("DELETE FROM reminders WHERE id=? AND code=?", (rid, c.code))
    return {"ok": True}


@app.post("/api/symptoms")
def add_symptom(b: SymptomIn, x_code: str = Header(None)):
    c = ctx_from(x_code)
    return Tools(c).log_symptom(b.symptom, b.severity, b.notes)


@app.get("/api/library")
def library(q: str = "", x_code: str = Header(None)):
    """Guideline library for the app's 'Guides' tab: search, or list all documents when q is empty."""
    ctx_from(x_code)
    from guidelines import index
    ix = index()
    if q.strip():
        return {"results": ix.search(q, 5)}
    docs = {}
    for p in ix.passages:
        docs.setdefault(p["doc"], {"title": p["doc"], "source": p["source"], "sections": []})["sections"].append(
            {"heading": p["section"], "text": p["text"]})
    return {"documents": list(docs.values())}


@app.post("/api/requests")
def add_request(b: RequestIn, x_code: str = Header(None)):
    c = ctx_from(x_code)
    return c.send_request(b.kind, b.text, b.urgency)


app.mount("/", StaticFiles(directory=os.path.join(APP_DIR, "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("LUNGCARE_PORT", "8100"))
    print(f"\n  LungCare patient app -> http://127.0.0.1:{port}\n  assistant: {brain.name}"
          f"{'' if brain.name == 'claude' else ' (set ANTHROPIC_API_KEY to use Claude)'}\n  hospital API: {DOCTOR_API}\n")
    host = os.environ.get("LUNGCARE_HOST", "127.0.0.1")
    if host == "0.0.0.0":
        import socket
        try:
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); sk.connect(("8.8.8.8", 80)); ip = sk.getsockname()[0]; sk.close()
        except OSError:
            ip = "this-PC-IP"
        print(f"  Android app: connect to  http://{ip}:{port}  (phone and PC on the same Wi-Fi)\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
