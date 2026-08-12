import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

"""
youtube.py
----------
Uses yt-dlp's Python API directly (no subprocess call to "yt-dlp") so it
works even when the yt-dlp script folder is not on the system PATH.

YouTube bot-detection workarounds applied:
  • Spoof a real Chrome browser User-Agent + headers
  • Use extractor_args to force the web_embedded player client
  • Automatic retry (3 attempts) across different format fallbacks
  • Accept cookies.txt from the project directory if present (optional)
"""

import subprocess
import tempfile
import time

from ffmpeg_utils import FFMPEG


# ── YouTube Download — via yt-dlp Python API ─────────────────────────────────
def download_youtube_video(url: str, output_dir: str = "/tmp") -> str | None:
    """
    Download YouTube video as MP4 using yt-dlp's Python API.
    Returns the local file path or None on failure.

    Applies several bot-detection workarounds so that most public videos
    can be downloaded without manual cookie export.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_template = os.path.join(output_dir, "yt_video.%(ext)s")

    # Look for an optional cookies.txt next to this file (Netscape format).
    # Users can export it from their browser with the "Get cookies.txt LOCALLY"
    # extension, then place it next to app.py to bypass age/region restrictions.
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    cookies_file = os.path.join(base_dir, "cookies.txt")

    CHROME_UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )

    # Format chains to try in order — from best quality to most permissive
    FORMAT_CHAINS = [
        # 1. Best video ≤480p + best audio, merged to mp4
        (
            "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]"
            "/bestvideo[height<=480]+bestaudio"
            "/best[height<=480]"
        ),
        # 2. Any mp4 up to 720p
        "best[height<=720][ext=mp4]/best[ext=mp4]",
        # 3. Absolute fallback — whatever yt-dlp can get
        "best",
    ]

    base_opts = {
        "outtmpl"             : out_template,
        "noplaylist"          : True,
        "socket_timeout"      : 60,
        "retries"             : 3,
        "fragment_retries"    : 3,
        "quiet"               : True,
        "no_warnings"         : True,
        "merge_output_format" : "mp4",
        "ffmpeg_location"     : os.path.dirname(FFMPEG),
        # Spoof a real browser
        "http_headers": {
            "User-Agent"     : CHROME_UA,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept"         : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        # Force yt-dlp to use the web_embedded player client — bypasses
        # the most common "Sign in to confirm you're not a bot" gate.
        "extractor_args": {
            "youtube": {
                "player_client": ["web_embedded", "web", "android"],
                "skip"         : ["dash", "hls"],
            }
        },
    }

    # Attach cookies if the file exists
    if os.path.exists(cookies_file):
        base_opts["cookiefile"] = cookies_file
        print(f"  [youtube] Using cookies from {cookies_file}")

    try:
        import yt_dlp
    except ImportError:
        print("  [youtube] yt-dlp not installed — run: pip install yt-dlp")
        return None

    last_error = None
    for i, fmt in enumerate(FORMAT_CHAINS):
        opts = {**base_opts, "format": fmt}
        try:
            print(f"  [youtube] Download attempt {i + 1}/{len(FORMAT_CHAINS)} "
                  f"(format: {fmt[:60]}...)")
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])

            # Check whether the file landed
            path = _find_output(output_dir)
            if path:
                print(f"  [youtube] Downloaded → {path}")
                return path

        except yt_dlp.utils.DownloadError as e:
            last_error = e
            print(f"  [youtube] Attempt {i + 1} failed: {e}")
            time.sleep(1)   # brief pause before retry
        except Exception as e:
            last_error = e
            print(f"  [youtube] Unexpected error on attempt {i + 1}: {e}")
            time.sleep(1)

    print(f"  [youtube] All download attempts failed. Last error: {last_error}")
    return None


def _find_output(output_dir: str) -> str | None:
    """Return the downloaded mp4 path, or any video file if mp4 is missing."""
    VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")
    # Prefer mp4
    for f in os.listdir(output_dir):
        if f.startswith("yt_video") and f.endswith(".mp4"):
            return os.path.join(output_dir, f)
    # Fallback to any video file
    for f in os.listdir(output_dir):
        if f.startswith("yt_video") and os.path.splitext(f)[1] in VIDEO_EXTS:
            return os.path.join(output_dir, f)
    return None


# ── Whisper ASR — tiny model for speed ───────────────────────────────────────
_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print("[whisper] Loading tiny model (fast)...")
        _whisper_model = whisper.load_model("tiny")
    return _whisper_model


def transcribe_audio(wav_path: str, language: str = "english") -> str:
    """
    Transcribe WAV using Whisper tiny model.
    Only processes first 60 seconds for speed.
    """
    lang_code = {"english": "en", "kannada": "kn"}.get(language.lower(), "en")

    # Trim to first 60 seconds to avoid processing long videos
    trimmed_path = wav_path.replace(".wav", "_trim.wav")
    trim_cmd = [
        FFMPEG, "-y", "-loglevel", "quiet",
        "-i", wav_path,
        "-t", "60",
        "-acodec", "copy",
        trimmed_path,
    ]
    try:
        subprocess.run(trim_cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=30)
    except Exception:
        pass  # trimming is optional; fall back to full audio

    audio_to_use = trimmed_path if os.path.exists(trimmed_path) else wav_path

    try:
        model  = _get_whisper()
        result = model.transcribe(
            audio_to_use,
            language                   = lang_code,
            fp16                       = False,
            verbose                    = False,
            condition_on_previous_text = False,
            temperature                = 0,
        )
        return result.get("text", "").strip()
    except Exception as e:
        print(f"  [whisper] Error: {e}")
        return ""
    finally:
        if os.path.exists(trimmed_path):
            try:
                os.remove(trimmed_path)
            except Exception:
                pass
