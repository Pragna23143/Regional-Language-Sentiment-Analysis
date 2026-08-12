"""
check_setup.py  —  run this first to diagnose your setup.

    python check_setup.py

Prints everything found / missing so you know exactly what to fix.
"""
import os, sys, shutil, subprocess

SEP = "=" * 60

print(SEP)
print("  Emotion Recognition v2 — Setup Checker")
print(SEP)

# ── Python ────────────────────────────────────────────────────────────────────
print(f"\n[Python] {sys.version.split()[0]}  at  {sys.executable}\n")

# ── ffmpeg_utils import ───────────────────────────────────────────────────────
print("── ffmpeg_utils.py ──────────────────────────────────────────")
try:
    import ffmpeg_utils as fu
    print(f"  FFMPEG  = {fu.FFMPEG}")
    print(f"  FFPROBE = {fu.FFPROBE}")
    exe_ok = os.path.isfile(fu.FFMPEG) or fu.FFMPEG == "ffmpeg"
    print(f"  exe exists on disk: {exe_ok}")
except Exception as e:
    print(f"  IMPORT ERROR: {e}")
    fu = None

# ── subprocess test ───────────────────────────────────────────────────────────
print("\n── subprocess test (ffmpeg -version) ────────────────────────")
ffmpeg_cmd = fu.FFMPEG if fu else "ffmpeg"
try:
    r = subprocess.run([ffmpeg_cmd, "-version"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    first = (r.stdout or r.stderr).decode(errors="replace").split("\n")[0]
    print(f"  [OK] {first}")
except FileNotFoundError:
    print(f"  [FAIL] FileNotFoundError — '{ffmpeg_cmd}' not found by subprocess")
    print(f"         Your PATH (first 500 chars): {os.environ.get('PATH','')[:500]}")
except Exception as e:
    print(f"  [FAIL] {e}")

# ── manual search ─────────────────────────────────────────────────────────────
print("\n── Manual search for ffmpeg.exe on this PC ─────────────────")
found = []
users = r"C:\Users"
if os.path.isdir(users):
    for u in os.listdir(users):
        dl = os.path.join(users, u, "Downloads")
        if not os.path.isdir(dl): continue
        for entry in os.listdir(dl):
            if entry.lower().startswith("ffmpeg"):
                folder = os.path.join(dl, entry)
                for root, dirs, files in os.walk(folder):
                    if "ffmpeg.exe" in files:
                        found.append(os.path.join(root, "ffmpeg.exe"))
if found:
    for p in found:
        print(f"  FOUND: {p}")
    print(f"\n  → To fix, run this once before starting the server:")
    print(f"    set FFMPEG_BIN_DIR={os.path.dirname(found[0])}")
else:
    print("  No ffmpeg.exe found in any C:\\Users\\*\\Downloads\\ffmpeg* folder.")
    print("  Make sure you extracted the ffmpeg zip file from gyan.dev or ffmpeg.org.")

# ── yt-dlp ────────────────────────────────────────────────────────────────────
print("\n── yt-dlp ───────────────────────────────────────────────────")
yt = shutil.which("yt-dlp")
if yt:
    print(f"  [OK] {yt}")
else:
    print("  [MISSING]  pip install yt-dlp")

# ── packages ──────────────────────────────────────────────────────────────────
print("\n── Python packages ──────────────────────────────────────────")
for mod, pkg in [("torch","torch"),("fastapi","fastapi"),("uvicorn","uvicorn"),
                 ("librosa","librosa"),("cv2","opencv-python"),
                 ("whisper","openai-whisper"),("yt_dlp","yt-dlp"),
                 ("jwt","PyJWT"),("google.oauth2","google-auth")]:
    try:
        __import__(mod); print(f"  [OK] {pkg}")
    except ImportError:
        print(f"  [MISSING] {pkg}  →  pip install {pkg}")

# ── checkpoints ───────────────────────────────────────────────────────────────
print("\n── Model checkpoints ────────────────────────────────────────")
base = os.path.dirname(os.path.abspath(__file__))
for f in ("audio_model.pth","video_model.pth"):
    p = os.path.join(base,"checkpoints",f)
    if os.path.isfile(p):
        print(f"  [OK] {f}  ({os.path.getsize(p)//1024} KB)")
    else:
        print(f"  [MISSING] checkpoints/{f}  → run train.py")

print(f"\n{SEP}")
print("  Done. Read the output above and fix any [FAIL] / [MISSING] items.")
print(SEP)
