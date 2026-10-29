"""Reset a forgotten password of the clinician workstation - run ON THE AI SERVER PC only
(physical access to this PC is what authorises the reset). Double-click RESET_PASSWORD.bat.

    python doctor_app/reset_password.py                 (asks for the user and the new password)
"""
import getpass
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auth  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(os.environ.get("LUNGSCAN_APP_DATA", os.path.join(ROOT, "app_data")), "lungscan.db")


def main():
    if not os.path.exists(DB):
        print("No doctor-app database found at", DB, "- start 7_DOCTOR_APP.bat once first.")
        return
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    users = con.execute("SELECT id, username, name, role, active FROM users ORDER BY id").fetchall()
    if not users:
        print("There are no accounts yet. Start 7_DOCTOR_APP.bat and create the administrator account.")
        return
    print("\nAccounts on this workstation:")
    for u in users:
        print(f"   {u['username']:<20} {u['name'] or '':<28} {u['role']}{'' if u['active'] else '  (deactivated)'}")
    name = input("\nUsername to reset: ").strip().lower()
    u = con.execute("SELECT * FROM users WHERE username=?", (name,)).fetchone()
    if u is None:
        print("No such user.")
        return
    while True:
        pw = getpass.getpass("New password (at least 8 characters, hidden while typing): ")
        if len(pw) < 8:
            print("Too short.")
            continue
        if getpass.getpass("Type it again: ") != pw:
            print("The two passwords differ, try again.")
            continue
        break
    con.execute("UPDATE users SET pw_hash=?, active=1 WHERE id=?", (auth.hash_password(pw), u["id"]))
    con.execute("DELETE FROM sessions WHERE user_id=?", (u["id"],))
    auth.audit(con, "server-pc", "password reset", f"{u['username']} (reset on the server PC)")
    con.commit()
    print(f"\nDone. {u['username']} can now sign in with the new password.")


if __name__ == "__main__":
    main()
