"""
ffmpeg_utils.py  —  finds ffmpeg.exe and sets PATH before any subprocess call.

Detection order:
  1. Hardcoded known path (your exact download location — works immediately)
  2. FFMPEG_BIN_DIR env var
  3. FFMPEG_PATH env var (full path to exe)
  4. os.walk search inside every C:\\Users\\*\\Downloads\\ffmpeg* folder
  5. Fixed common locations (C:\\ffmpeg, Program Files, choco, scoop)
  6. System PATH (shutil.which)

Exports: FFMPEG, FFPROBE  — absolute paths used in every subprocess call.
"""

import os
import sys
import shutil


# ── 1. HARDCODED KNOWN LOCATION — edit this if your path is different ─────────
#    This is your exact ffmpeg download path.  If it exists, it is used first.
_HARDCODED = r"C:\Users\Pragna\Downloads\ffmpeg-9.0-essentials_build\ffmpeg-9.0-essentials_build\bin"


def _walk_find(folder: str) -> str | None:
    """Walk *folder* and return the first directory that contains ffmpeg.exe."""
    try:
        for root, dirs, files in os.walk(folder):
            if "ffmpeg.exe" in files:
                return root
    except Exception:
        pass
    return None


def _find_ffmpeg_dir() -> str | None:
    # 1. Hardcoded path (fastest — no search needed)
    if os.path.isfile(os.path.join(_HARDCODED, "ffmpeg.exe")):
        return _HARDCODED

    # 2. Env var: directory
    env_dir = os.environ.get("FFMPEG_BIN_DIR", "").strip()
    if env_dir:
        if os.path.isfile(os.path.join(env_dir, "ffmpeg.exe")):
            return env_dir
        # May point to exe directly
        if os.path.isfile(env_dir):
            return os.path.dirname(env_dir)

    # 3. Env var: full exe path
    env_exe = os.environ.get("FFMPEG_PATH", "").strip()
    if env_exe and os.path.isfile(env_exe):
        return os.path.dirname(env_exe)

    if sys.platform != "win32":
        return None   # Linux/macOS: rely on system PATH

    # 4. Walk every user's Downloads\ffmpeg* folder
    users_dir = r"C:\Users"
    if os.path.isdir(users_dir):
        for user in os.listdir(users_dir):
            downloads = os.path.join(users_dir, user, "Downloads")
            if not os.path.isdir(downloads):
                continue
            try:
                for entry in os.listdir(downloads):
                    if entry.lower().startswith("ffmpeg"):
                        candidate = os.path.join(downloads, entry)
                        if os.path.isdir(candidate):
                            found = _walk_find(candidate)
                            if found:
                                return found
            except Exception:
                pass

    # 5. Fixed common locations
    for path in [
        r"C:\ffmpeg\bin",
        r"C:\ffmpeg",
        r"C:\Program Files\ffmpeg\bin",
        r"C:\Program Files (x86)\ffmpeg\bin",
        r"C:\tools\ffmpeg\bin",
        r"C:\tools\ffmpeg",
        os.path.expandvars(r"%USERPROFILE%\scoop\apps\ffmpeg\current\bin"),
        r"C:\ProgramData\chocolatey\bin",
    ]:
        if os.path.isfile(os.path.join(path, "ffmpeg.exe")):
            return path

    # 6. System PATH
    if shutil.which("ffmpeg"):
        return None   # already reachable — use bare name

    return None   # truly not found


# ── Run once at import time ────────────────────────────────────────────────────
_FFMPEG_DIR = _find_ffmpeg_dir()

if _FFMPEG_DIR:
    _cur = os.environ.get("PATH", "")
    if _FFMPEG_DIR not in _cur:
        os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + _cur
    print(f"[ffmpeg_utils] ffmpeg found -> {_FFMPEG_DIR}")
    FFMPEG  = os.path.join(_FFMPEG_DIR, "ffmpeg.exe"  if sys.platform == "win32" else "ffmpeg")
    FFPROBE = os.path.join(_FFMPEG_DIR, "ffprobe.exe" if sys.platform == "win32" else "ffprobe")
else:
    if shutil.which("ffmpeg"):
        print("[ffmpeg_utils] ffmpeg is on system PATH.")
    else:
        print(
            "\n[ffmpeg_utils] ERROR: ffmpeg NOT FOUND.\n"
            "  Run check_setup.py for a full diagnosis, or:\n"
            "  1) Double-click run_server.bat  (sets PATH automatically)\n"
            "  2) OR set env var:  set FFMPEG_BIN_DIR=C:\\path\\to\\ffmpeg\\bin\n"
        )
    FFMPEG  = "ffmpeg"
    FFPROBE = "ffprobe"
