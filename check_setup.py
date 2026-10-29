"""Step 0 - check the PC, the GPU and the dataset folder BEFORE anything else.

    python check_setup.py

Copy everything it prints and send it back. It tells us whether the GPU works with your
PyTorch build, which batch size fits in GPU memory, how long one epoch will take, and whether
the LUNA16 folder is complete.
"""
import glob
import os
import platform
import shutil
import sys
import time

import config as C


def _bench(bs):
    """Child process: one clean GPU test for one batch size -> prints one RESULT json line."""
    import json
    import torch
    from model import MSGFNet
    dev = torch.device("cuda")
    try:
        torch.backends.cudnn.benchmark = True
        m = MSGFNet(width=C.TRAIN["width"]).to(dev)
        opt = torch.optim.AdamW(m.parameters(), 1e-3)
        loc = torch.randn(bs, 3, 32, 32, 32, device=dev)
        ctx = torch.randn(bs, 3, 48, 48, 48, device=dev)
        for i in range(6):
            if i == 2:
                torch.cuda.synchronize()
                t = time.time()
            with torch.autocast("cuda", enabled=C.amp_enabled(dev)):
                out = m(loc, ctx)
            loss = out["logit"].float().mean() + out["diameter"].float().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        print("RESULT " + json.dumps({"ok": True, "step_s": (time.time() - t) / 4,
                                      "peak_gb": torch.cuda.max_memory_allocated() / 1e9}))
    except Exception as e:
        print("RESULT " + json.dumps({"ok": False, "error": str(e).splitlines()[0][:150]}))


if len(sys.argv) == 3 and sys.argv[1] == "--bench-bs":
    _bench(int(sys.argv[2]))
    sys.exit(0)

OK, BAD, WARN = "[OK]  ", "[FAIL]", "[WARN]"
problems = []


def say(tag, msg):
    print(tag, msg)
    if tag == BAD:
        problems.append(msg)


print("=" * 70)
print("1) SOFTWARE")
print("=" * 70)
say(OK, f"Python {sys.version.split()[0]} on {platform.system()} {platform.release()}")
for mod in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "SimpleITK", "torch"):
    try:
        m = __import__(mod)
        say(OK, f"{mod} {getattr(m, '__version__', '')}".strip())
    except Exception as e:
        say(BAD, f"{mod} not importable ({e}) -> pip install -r requirements.txt")

print("\n" + "=" * 70)
print("2) GPU")
print("=" * 70)
gpu_ok = False
try:
    import torch
    say(OK, f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        say(BAD, "torch cannot see a GPU. Check that 'nvidia-smi' works in a terminal, then double-click 1B_ONLY_IF_GPU_FAILS.bat")
    else:
        dev = torch.device("cuda")
        p = torch.cuda.get_device_properties(0)
        cc = torch.cuda.get_device_capability(0)
        say(OK, f"GPU: {p.name}, {p.total_memory / 1e9:.1f} GB, compute capability {cc[0]}.{cc[1]}")
        say(OK, f"mixed precision will be {'ON' if C.amp_enabled(dev) else 'OFF (FP32 - correct for Pascal cards)'}")
        try:
            x = torch.randn(2, 3, 16, 16, 16, device=dev)
            torch.nn.Conv3d(3, 4, 3).to(dev)(x).sum().item()
            gpu_ok = True
            say(OK, "3D convolution runs on the GPU")
        except Exception as e:
            say(BAD, f"GPU kernel failed: {e}\n       -> this PyTorch build does not support your GPU. "
                     "Double-click 1B_ONLY_IF_GPU_FAILS.bat, then run 2_CHECK.bat again")
except Exception as e:
    say(BAD, f"torch problem: {e}")

print("\n" + "=" * 70)
print("3) DATASET FOLDER:", os.path.abspath(C.LUNA16_ROOT))
print("=" * 70)
n_mhd = 0
if not os.path.isdir(C.LUNA16_ROOT):
    say(BAD, "folder does not exist - set LUNA16_ROOT or edit config.py")
else:
    zips = glob.glob(os.path.join(C.LUNA16_ROOT, "*.zip")) + glob.glob(os.path.join(C.LUNA16_ROOT, "*.7z"))
    for s in range(10):
        mhds = glob.glob(os.path.join(C.LUNA16_ROOT, f"subset{s}", "**", "*.mhd"), recursive=True)
        missing_raw = [m for m in mhds if not os.path.exists(m[:-4] + ".raw")]
        n_mhd += len(mhds)
        if not mhds:
            say(BAD, f"subset{s}: no .mhd files (not extracted yet?)")
        elif missing_raw:
            say(BAD, f"subset{s}: {len(mhds)} .mhd but {len(missing_raw)} have no .raw (incomplete extraction)")
        else:
            say(OK, f"subset{s}: {len(mhds)} scans")
    say(OK if n_mhd == 888 else WARN, f"total scans: {n_mhd} (full LUNA16 = 888)")
    if zips:
        say(WARN, f"{len(zips)} archive(s) still in the folder: {[os.path.basename(z) for z in zips][:6]} "
                  "- fine if they are already extracted")
    for name, path, need in (("candidates", C.CANDIDATES_CSV, True), ("annotations", C.ANNOTATIONS_CSV, True),
                             ("annotations_excluded", C.EXCLUDED_CSV, False)):
        if os.path.exists(path):
            with open(path) as f:
                rows = sum(1 for _ in f) - 1
            say(OK, f"{os.path.basename(path)}: {rows} rows")
        else:
            say(BAD if need else WARN, f"{os.path.basename(path)} not found at {path}"
                + ("" if need else " (optional; from evaluationScript.zip -> annotations/)"))
    if os.path.exists(C.CANDIDATES_CSV) and os.path.basename(C.CANDIDATES_CSV) != "candidates_V2.csv":
        say(WARN, "using the old candidates.csv; candidates_V2.csv is preferred")

print("\n" + "=" * 70)
print("4) DISK AND MEMORY")
print("=" * 70)
os.makedirs(C.WORK_DIR, exist_ok=True)
free = shutil.disk_usage(os.path.abspath(C.WORK_DIR)).free / 1e9
say(OK if free > 15 else BAD, f"free space for WORK_DIR ({os.path.abspath(C.WORK_DIR)}): {free:.0f} GB (need ~15 GB)")
try:
    import psutil
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    say(OK if vm.available > 8e9 else BAD,
        f"RAM {vm.total / 1e9:.0f} GB, AVAILABLE now {vm.available / 1e9:.1f} GB, "
        f"page file {sw.total / 1e9:.0f} GB, CPU cores {os.cpu_count()}"
        + ("" if vm.available > 8e9 else "  <- too little free memory: close other programs or restart the PC"))
except ImportError:
    say(OK, f"CPU cores {os.cpu_count()} (pip install psutil to also see RAM)")

if n_mhd:
    print("\n" + "=" * 70)
    print("5) ONE SCAN (reading + patch extraction speed)")
    print("=" * 70)
    try:
        from luna_io import Scan
        mhd = sorted(glob.glob(os.path.join(C.LUNA16_ROOT, "subset*", "**", "*.mhd"), recursive=True))[0]
        t = time.time()
        sc = Scan(mhd)
        t_read = time.time() - t
        t = time.time()
        centre = sc.index_xyz_to_world(sc.spacing * 0 + [sc.array.shape[2] / 2, sc.array.shape[1] / 2, sc.array.shape[0] / 2])
        for _ in range(10):
            sc.extract_patch(centre)
        t_p = (time.time() - t) / 10
        say(OK, f"scan {sc.array.shape} (z,y,x), spacing {tuple(round(float(v), 2) for v in sc.spacing)} mm: "
                f"read {t_read:.1f} s, {t_p * 1000:.0f} ms per patch")
    except Exception as e:
        say(BAD, f"could not read a scan: {e}")

if gpu_ok:
    print("\n" + "=" * 70)
    print("6) TRAINING SPEED / MEMORY ON THIS GPU (each test in a fresh process)")
    print("=" * 70)
    import json
    import subprocess
    import torch
    free, total = torch.cuda.mem_get_info()
    used_by_others = (total - free) / 1e9
    say(OK if used_by_others < 1.5 else WARN,
        f"GPU memory free now: {free / 1e9:.1f} of {total / 1e9:.1f} GB"
        + ("" if used_by_others < 1.5 else f"  <- {used_by_others:.1f} GB used by OTHER programs "
           "(close them / ask if someone else is using this GPU)"))
    best = None
    for bs in (8, 16, 24, 32, 48, 64):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--bench-bs", str(bs)],
                           capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        res = json.loads(line[-1][7:]) if line else {"ok": False, "error": (r.stderr or "crashed").strip().splitlines()[-1][:150]}
        if not res["ok"]:
            say(WARN if best else BAD, f"batch {bs}: does not fit ({res['error']})")
            break
        epoch_min = res["step_s"] * C.TRAIN["samples_per_epoch"] / bs / 60
        say(OK, f"batch {bs}: {res['step_s']:.2f} s/step, peak {res['peak_gb']:.1f} GB, ~{epoch_min:.1f} min/epoch")
        if res["peak_gb"] < 0.8 * total / 1e9:
            best = (bs, epoch_min)
    if best:
        with open(C.LOCAL_SETTINGS, "w") as f:
            json.dump({"batch_size": best[0]}, f)
        say(OK, f"-> training will use batch size {best[0]} automatically "
                f"(~{best[1]:.0f} min/epoch, ~{best[1] * C.TRAIN['epochs'] / 60:.1f} h for {C.TRAIN['epochs']} epochs)")

print("\n" + "=" * 70)
print("RESULT:", "READY - go to the next step" if not problems else f"{len(problems)} problem(s) to fix:")
for p in problems:
    print("   -", p)
print("=" * 70)
