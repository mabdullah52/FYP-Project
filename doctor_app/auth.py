"""Accounts, sessions, settings and audit log for the clinician workstation.

* roles: "doctor" (clinical work) and "admin" (also manages users, AI thresholds, logs, records)
* passwords: PBKDF2-SHA256, 200k iterations, per-user salt; sessions: random token in an HttpOnly cookie
* first start: no users -> the web page shows "create the administrator account"
"""
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta

COOKIE = "ls_session"
SESSION_HOURS = 12
DEFAULT_SETTINGS = {
    "detection_threshold": 0.5,     # nodule probability above which a candidate is reported
    "review_uncertainty": 0.15,     # MC/TTA std above which a finding is flagged "needs review"
    "slice_yellow": 0.30,           # slice-level risk bar: below = green, between = yellow
    "slice_red": 0.70,              # above = red  (thresholds from the project proposal)
    "max_findings": 10,
    "fast_mode": 1,                 # 1 = quick pass + full TTA/MC on top candidates, 0 = full TTA/MC on all
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, name TEXT, role TEXT NOT NULL,
    pw_hash TEXT NOT NULL, active INTEGER DEFAULT 1, created TEXT, last_login TEXT);
CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, user_id INTEGER, created TEXT, expires TEXT);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS audit_log(
    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, user TEXT, action TEXT, detail TEXT);
"""


def now():
    return datetime.now().isoformat(timespec="seconds")


def hash_password(pw, salt=None):
    salt = salt or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()
    return f"pbkdf2${salt}${h}"


def check_password(pw, stored):
    try:
        _, salt, h = stored.split("$")
    except ValueError:
        return False
    return hmac.compare_digest(hash_password(pw, salt).split("$")[2], h)


def init(con):
    con.executescript(SCHEMA)
    for k, v in DEFAULT_SETTINGS.items():
        con.execute("INSERT OR IGNORE INTO settings VALUES(?,?)", (k, json.dumps(v)))


def get_settings(con):
    out = dict(DEFAULT_SETTINGS)
    for r in con.execute("SELECT key, value FROM settings"):
        out[r["key"]] = json.loads(r["value"])
    return out


def set_settings(con, values):
    for k, v in values.items():
        if k in DEFAULT_SETTINGS:
            con.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (k, json.dumps(v)))


def n_users(con):
    return con.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def create_user(con, username, name, role, password):
    con.execute("INSERT INTO users(username, name, role, pw_hash, created) VALUES(?,?,?,?,?)",
                (username.strip().lower(), name.strip() or username, role, hash_password(password), now()))


def login(con, username, password):
    u = con.execute("SELECT * FROM users WHERE username=?", (username.strip().lower(),)).fetchone()
    if u is None or not u["active"] or not check_password(password, u["pw_hash"]):
        return None
    token = secrets.token_urlsafe(32)
    con.execute("INSERT INTO sessions VALUES(?,?,?,?)",
                (token, u["id"], now(), (datetime.now() + timedelta(hours=SESSION_HOURS)).isoformat(timespec="seconds")))
    con.execute("UPDATE users SET last_login=? WHERE id=?", (now(), u["id"]))
    return token


def user_for(con, token):
    if not token:
        return None
    r = con.execute("SELECT u.id, u.username, u.name, u.role, u.active, s.expires FROM sessions s "
                    "JOIN users u ON u.id = s.user_id WHERE s.token=?", (token,)).fetchone()
    if r is None or not r["active"] or r["expires"] < now():
        return None
    return {"id": r["id"], "username": r["username"], "name": r["name"], "role": r["role"]}


def logout(con, token):
    con.execute("DELETE FROM sessions WHERE token=?", (token,))


def audit(con, user, action, detail=""):
    con.execute("INSERT INTO audit_log(at, user, action, detail) VALUES(?,?,?,?)",
                (now(), (user or {}).get("username", "system") if isinstance(user, dict) else (user or "system"),
                 action, str(detail)[:500]))
