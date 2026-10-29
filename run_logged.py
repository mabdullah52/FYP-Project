"""Run a script, show its output on screen AND save it to logs/<name>.txt.
Used by the numbered .bat files:  python run_logged.py <logname> <script.py> [args...]"""
import os
import subprocess
import sys

log_name, cmd = sys.argv[1], sys.argv[2:]
os.makedirs("logs", exist_ok=True)
path = os.path.join("logs", log_name + ".txt")
env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
with open(path, "a", encoding="utf-8") as log:
    log.write("\n===== " + " ".join(cmd) + " =====\n")
    p = subprocess.Popen([sys.executable] + cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    for raw in p.stdout:
        line = raw.decode("utf-8", errors="replace")
        sys.stdout.write(line)
        sys.stdout.flush()
        log.write(line)
        log.flush()
    code = p.wait()
    if code != 0:
        msg = {3221225477: "crashed (access violation) - usually the PC ran out of memory",
               3221225725: "crashed (stack overflow)", 3221226505: "crashed (Windows fast-fail)"}.get(code % 2 ** 32, "")
        line = f"!!! exit code {code} {msg}\n"
        sys.stdout.write(line); log.write(line)
print(f"\n(saved to {path})")
sys.exit(code)
