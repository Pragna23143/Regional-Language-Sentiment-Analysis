import os, sys

# ── Load .env file if present (for local development) ────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass  # python-dotenv not installed; rely on real env vars

# ── Auto-locate ffmpeg/ffprobe (Windows + Linux/macOS) ───────────────────────
import ffmpeg_utils   # noqa: F401  (side-effect: sets PATH + exports FFMPEG/FFPROBE)

"""
app.py
------
FastAPI backend for Emotion Recognition — Emotion & Intent Detection API.

Endpoints:
  POST /predict                — accepts YouTube URL, returns full analysis
  POST /predict/upload         — accepts uploaded video file
  POST /predict/timeline       — returns emotion timeline across the whole video
  GET  /health                 — health check

  POST /auth/signup            — create account (name, email, password)
  POST /auth/login             — login (email, password) → JWT
  GET  /auth/me                — return current user (requires JWT)

  GET  /history                — logged-in user's past analyses
  DELETE /history/{id}         — delete a history entry

  POST /chat                   — GPT-powered Q&A chatbot about the analysed video

Run:
  uvicorn app:app --reload --port 8000
"""

import hashlib
import secrets
import shutil
import sqlite3
import tempfile
import time
import traceback
from typing import Optional, List

import jwt                   # PyJWT
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from model import AudioCNNLSTM, VideoCNN, EMOTION_CLASSES
from audio import N_MFCC
from youtube import download_youtube_video
from test import predict, load_models, predict_timeline

# ── Config ────────────────────────────────────────────────────────────────────
JWT_SECRET    = os.environ.get("JWT_SECRET", "affectiq-dev-secret-do-not-share")
JWT_ALGORITHM = "HS256"
JWT_EXP_SECS  = 60 * 60 * 24 * 7   # 7 days

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

# The DB used to live inside BASE_DIR (next to index.html). If you serve the
# frontend with a dev tool like VS Code "Live Server", it watches that whole
# folder and auto-reloads the browser tab on ANY file change inside it —
# including every write to this database, which happens right after each
# successful analysis. That caused the page to flash the results and
# immediately reload/reset. Fix: keep the DB outside the served folder.
_DATA_DIR = os.path.join(tempfile.gettempdir(), "emotion_recognition_data")
os.makedirs(_DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(_DATA_DIR, "affectiq.db")

_OLD_DB_PATH = os.path.join(BASE_DIR, "affectiq.db")
if os.path.exists(_OLD_DB_PATH) and not os.path.exists(DB_PATH):
    # One-time migration so existing signups/history aren't lost.
    shutil.copy2(_OLD_DB_PATH, DB_PATH)
    print(f"[API] Migrated existing database from {_OLD_DB_PATH} to {DB_PATH}")


def _get_openai_key() -> str:
    """Read OPENAI_API_KEY at call-time so .env changes take effect without restart."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        # Try re-loading .env in case it was added after startup
        try:
            from dotenv import load_dotenv
            load_dotenv(ENV_PATH, override=True)
            key = os.environ.get("OPENAI_API_KEY", "")
        except Exception:
            pass
    return key


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title      = "Emotion Recognition",
    description= "Multimodal emotion and intent detection from YouTube videos",
    version    = "4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

security = HTTPBearer(auto_error=False)


# ── Password helpers (stdlib only — no extra packages) ───────────────────────
def _hash_password(password: str, salt: str | None = None) -> str:
    """Return 'salt:hash' string using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return f"{salt}:{dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    """Verify password against stored 'salt:hash'."""
    try:
        salt, _ = stored.split(":", 1)
        return secrets.compare_digest(stored, _hash_password(password, salt))
    except Exception:
        return False


# ── Database helpers ──────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL,
            email      TEXT    UNIQUE NOT NULL,
            password   TEXT    NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            url         TEXT    NOT NULL,
            emotion     TEXT,
            intent      TEXT,
            full_result TEXT,
            created_at  INTEGER NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    # Migrate existing DBs that don't have the full_result column yet
    try:
        c.execute("ALTER TABLE history ADD COLUMN full_result TEXT")
        conn.commit()
    except Exception:
        pass  # column already exists
    conn.commit()
    conn.close()


# ── Auth helpers ──────────────────────────────────────────────────────────────
def _make_jwt(user_id: int, email: str) -> str:
    payload = {
        "sub"  : str(user_id),
        "email": email,
        "iat"  : int(time.time()),
        "exp"  : int(time.time()) + JWT_EXP_SECS,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Optional[dict]:
    if credentials is None:
        return None
    try:
        return jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except Exception:
        return None


def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    user = get_current_user(credentials)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


# ── History helper ────────────────────────────────────────────────────────────
import json as _json_mod

def _save_history(user_id: int, url: str, emotion: str, intent: str, full_result: dict = None):
    conn = get_db()
    conn.execute(
        "INSERT INTO history (user_id, url, emotion, intent, full_result, created_at) VALUES (?,?,?,?,?,?)",
        (user_id, url, emotion, intent, _json_mod.dumps(full_result) if full_result else None, int(time.time())),
    )
    conn.commit()
    conn.close()


# ── Load models once at startup ───────────────────────────────────────────────
audio_model = None
video_model = None


@app.on_event("startup")
async def startup_event():
    global audio_model, video_model
    init_db()
    try:
        audio_model, video_model = load_models()
        print("[API] Models loaded successfully.")
    except FileNotFoundError as e:
        print(f"[API] WARNING: {e}")
        print("[API] Run train.py first. Predictions will use dummy output.")

    if _get_openai_key():
        print("[API] OpenAI API key detected — GPT intent analysis and chatbot enabled.")
    else:
        print("[API] WARNING: OPENAI_API_KEY not set — add it to .env to enable GPT features.")


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    url     : str
    language: Optional[str] = "english"


class PredictResponse(BaseModel):
    emotion           : str
    intent            : str
    intent_explanation: Optional[str] = None
    language          : str
    transcript        : str
    audio_emotion     : str
    video_emotion     : str
    audio_probs       : dict
    video_probs       : dict
    intent_probs      : dict
    audio_energy      : float


class TimelineRequest(BaseModel):
    url         : str
    language    : Optional[str] = "english"
    window_secs : Optional[float] = 8.0


class SignupRequest(BaseModel):
    name    : str
    email   : str
    password: str


class LoginRequest(BaseModel):
    email   : str
    password: str


class ChatMessage(BaseModel):
    role   : str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message : str
    context : Optional[dict] = None   # last analysis result (full_result)
    history : Optional[List[ChatMessage]] = None  # conversation so far


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status"        : "ok",
        "models_loaded" : audio_model is not None,
        "emotions"      : EMOTION_CLASSES,
        "openai_enabled": bool(_get_openai_key()),
    }


# ── Auth endpoints ────────────────────────────────────────────────────────────
@app.post("/auth/signup")
async def auth_signup(req: SignupRequest):
    name     = req.name.strip()
    email    = req.email.strip().lower()
    password = req.password

    if not name or not email or not password:
        raise HTTPException(400, "name, email and password are required")
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")

    conn = get_db()
    c    = conn.cursor()
    c.execute("SELECT id FROM users WHERE email = ?", (email,))
    if c.fetchone():
        conn.close()
        raise HTTPException(409, "An account with that email already exists")

    hashed = _hash_password(password)
    c.execute(
        "INSERT INTO users (name, email, password, created_at) VALUES (?,?,?,?)",
        (name, email, hashed, int(time.time())),
    )
    conn.commit()
    user_id = c.lastrowid
    conn.close()

    token = _make_jwt(user_id, email)
    return {
        "token": token,
        "user" : {"id": user_id, "name": name, "email": email},
    }


@app.post("/auth/login")
async def auth_login(req: LoginRequest):
    email    = req.email.strip().lower()
    password = req.password

    conn = get_db()
    c    = conn.cursor()
    c.execute("SELECT id, name, email, password FROM users WHERE email = ?", (email,))
    row  = c.fetchone()
    conn.close()

    if not row or not _verify_password(password, row["password"]):
        raise HTTPException(401, "Invalid email or password")

    token = _make_jwt(row["id"], row["email"])
    return {
        "token": token,
        "user" : {"id": row["id"], "name": row["name"], "email": row["email"]},
    }


@app.get("/auth/me")
async def auth_me(user: dict = Depends(require_auth)):
    conn = get_db()
    c    = conn.cursor()
    c.execute("SELECT id, name, email FROM users WHERE id = ?", (int(user["sub"]),))
    row  = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "User not found")
    return dict(row)


# ── History endpoints ─────────────────────────────────────────────────────────
@app.get("/history")
async def get_history(user: dict = Depends(require_auth)):
    conn = get_db()
    c    = conn.cursor()
    c.execute(
        "SELECT id, url, emotion, intent, full_result, created_at FROM history "
        "WHERE user_id = ? ORDER BY created_at DESC LIMIT 50",
        (int(user["sub"]),),
    )
    rows = []
    for r in c.fetchall():
        row = dict(r)
        if row.get("full_result"):
            try:
                row["full_result"] = _json_mod.loads(row["full_result"])
            except Exception:
                row["full_result"] = None
        rows.append(row)
    conn.close()
    return {"history": rows}


@app.delete("/history/{entry_id}")
async def delete_history(entry_id: int, user: dict = Depends(require_auth)):
    conn = get_db()
    conn.execute(
        "DELETE FROM history WHERE id = ? AND user_id = ?",
        (entry_id, int(user["sub"])),
    )
    conn.commit()
    conn.close()
    return {"deleted": entry_id}


# ── Prediction endpoints ───────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict_from_url(
    req : PredictRequest,
    user: Optional[dict] = Depends(get_current_user),
):
    if req.language not in ("english", "kannada"):
        raise HTTPException(400, "language must be 'english' or 'kannada'")

    tmp_dir = tempfile.mkdtemp(prefix="eid_")
    try:
        print(f"[predict] Downloading: {req.url}")
        video_path = download_youtube_video(req.url, output_dir=tmp_dir)
        if not video_path:
            raise HTTPException(422,
                "Could not download the YouTube video. "
                "Possible causes:\n"
                "1. The video is age-restricted or private — export cookies.txt from your browser "
                "and place it next to app.py.\n"
                "2. yt-dlp is outdated — run: pip install -U yt-dlp\n"
                "3. The URL is invalid — paste the full YouTube URL and try again.")

        print(f"[predict] Running inference on: {video_path}")
        result = predict(
            video_path  = video_path,
            language    = req.language,
            audio_model = audio_model,
            video_model = video_model,
        )

        if user:
            try:
                _save_history(
                    user_id     = int(user["sub"]),
                    url         = req.url,
                    emotion     = result["emotion"],
                    intent      = result["intent"],
                    full_result = result,
                )
            except Exception as he:
                print(f"[WARN] /predict history save failed (non-fatal): {he}")

        return PredictResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        print("\n[ERROR] /predict failed:")
        traceback.print_exc()
        raise HTTPException(500, f"Prediction failed: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/predict/upload", response_model=PredictResponse)
async def predict_from_upload(
    file    : UploadFile      = File(...),
    language: str             = Form("english"),
    user    : Optional[dict]  = Depends(get_current_user),
):
    if language not in ("english", "kannada"):
        raise HTTPException(400, "language must be 'english' or 'kannada'")

    tmp_dir  = tempfile.mkdtemp(prefix="eid_upload_")
    ext      = os.path.splitext(file.filename)[1] or ".mp4"
    vid_path = os.path.join(tmp_dir, f"upload{ext}")

    try:
        with open(vid_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        print(f"[predict/upload] Running inference on: {vid_path}")
        result = predict(
            video_path  = vid_path,
            language    = language,
            audio_model = audio_model,
            video_model = video_model,
        )

        if user:
            try:
                _save_history(
                    user_id     = int(user["sub"]),
                    url         = f"[upload] {file.filename}",
                    emotion     = result["emotion"],
                    intent      = result["intent"],
                    full_result = result,
                )
            except Exception as he:
                print(f"[WARN] /predict/upload history save failed (non-fatal): {he}")

        return PredictResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        print("\n[ERROR] /predict/upload failed:")
        traceback.print_exc()
        raise HTTPException(500, f"Prediction failed: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/predict/timeline")
async def predict_timeline_endpoint(
    req : TimelineRequest,
    user: Optional[dict] = Depends(get_current_user),
):
    if req.language not in ("english", "kannada"):
        raise HTTPException(400, "language must be 'english' or 'kannada'")

    window = max(4.0, min(float(req.window_secs or 8.0), 30.0))

    tmp_dir = tempfile.mkdtemp(prefix="eid_tl_")
    try:
        video_path = download_youtube_video(req.url, output_dir=tmp_dir)
        if not video_path:
            raise HTTPException(422, "Could not download the YouTube video.")

        import subprocess as _sp
        import json as _json
        from ffmpeg_utils import FFPROBE as _FFPROBE
        probe = _sp.run(
            [_FFPROBE, "-v", "quiet", "-print_format", "json",
             "-show_format", video_path],
            stdout=_sp.PIPE, stderr=_sp.DEVNULL, timeout=15,
        )
        dur = float(_json.loads(probe.stdout).get("format", {}).get("duration", 0))

        # FFprobe can fail on some downloaded containers.  Fall back to
        # OpenCV metadata before running the timeline.
        if dur <= 0:
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                    if fps > 0 and frames > 0:
                        dur = frames / fps
                cap.release()
            except Exception:
                pass

        timeline = predict_timeline(
            video_path  = video_path,
            language    = req.language,
            audio_model = audio_model,
            video_model = video_model,
            window_secs = window,
            stride_secs = window,
        )

        if user:
            try:
                # Build a lightweight full_result (averaged probs across all
                # windows) so a saved timeline entry shows real data instead
                # of blank placeholders when reopened later.
                avg_probs = {e: 0.0 for e in EMOTION_CLASSES}
                for w in timeline:
                    for e in EMOTION_CLASSES:
                        avg_probs[e] += (w.get("fused_probs") or {}).get(e, 0.0)
                if timeline:
                    avg_probs = {e: round(v / len(timeline), 4) for e, v in avg_probs.items()}
                top_emotion = timeline[0]["fused_emotion"] if timeline else "neutral"

                _save_history(
                    user_id     = int(user["sub"]),
                    url         = req.url,
                    emotion     = top_emotion,
                    intent      = "timeline",
                    full_result = {
                        "emotion"            : top_emotion,
                        "intent"             : "timeline",
                        "intent_explanation" : "Intent analysis is not available in timeline mode.",
                        "language"           : req.language,
                        "transcript"         : f"(Timeline mode — {len(timeline)} windows analysed over {round(dur, 2)}s)",
                        "audio_emotion"      : top_emotion,
                        "video_emotion"      : top_emotion,
                        "audio_probs"        : avg_probs,
                        "video_probs"        : avg_probs,
                        "intent_probs"       : {},
                        "audio_energy"       : 0.0,
                    },
                )
            except Exception as he:
                print(f"[WARN] /predict/timeline history save failed (non-fatal): {he}")

        if not timeline:
            raise RuntimeError(
                "No timeline windows were produced. The video could not be decoded."
            )

        return {
            "duration": round(dur, 2),
            "window_secs": window,
            "timeline": timeline,
        }

    except HTTPException:
        raise
    except Exception as e:
        print("\n[ERROR] /predict/timeline failed:")
        traceback.print_exc()
        raise HTTPException(500, f"Timeline prediction failed: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Chat endpoint ─────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat_about_video(
    req : ChatRequest,
    user: dict = Depends(require_auth),
):
    """
    GPT-4o-mini powered chatbot for Q&A about the analysed video.
    Requires authentication and an OpenAI API key.
    """
    # Read the key at call-time so .env changes take effect without restart
    api_key = _get_openai_key()

    if not api_key:
        raise HTTPException(
            503,
            "OpenAI API key not configured.\n\n"
            "Create a file called .env next to app.py and add:\n"
            "  OPENAI_API_KEY=sk-...\n"
            "Then restart the server (Ctrl+C → uvicorn app:app --reload --port 8000)."
        )

    try:
        import openai
    except ImportError:
        raise HTTPException(503, "openai package not installed. Run: pip install openai>=1.30.0")

    client = openai.OpenAI(api_key=api_key)

    # Build system prompt with analysis context if provided
    system_parts = [
        "You are Emotion Recognition's intelligent video analysis assistant. "
        "You help users understand the emotional and intent analysis of their videos. "
        "Be concise, insightful, and friendly."
    ]

    if req.context:
        ctx = req.context
        transcript = (ctx.get("transcript") or "").strip()
        system_parts.append(f"""
=== Video Analysis Results ===
Detected Emotion   : {ctx.get("emotion", "unknown")}
Audio Emotion      : {ctx.get("audio_emotion", "unknown")}
Visual Emotion     : {ctx.get("video_emotion", "unknown")}
Detected Intent    : {ctx.get("intent", "unknown")}
Language           : {ctx.get("language", "unknown")}
Audio Energy       : {ctx.get("audio_energy", "unknown")}

Audio Emotion Probs: {ctx.get("audio_probs", {})}
Video Emotion Probs: {ctx.get("video_probs", {})}
Intent Probs       : {ctx.get("intent_probs", {})}

Intent Explanation : {ctx.get("intent_explanation", "(not available)")}

Transcript (first 3000 chars):
{transcript[:3000] if transcript else "(no transcript available)"}
==============================

Answer the user's questions based on this analysis. Be specific and reference actual data from the results above when relevant.
If the user asks something beyond the scope of the analysis (e.g. general questions), you can answer from general knowledge but note it's not from the video data.""")
    else:
        system_parts.append(
            "\nNo video has been analysed yet. Ask the user to run an analysis first, "
            "then they can ask questions about the results."
        )

    system_prompt = "\n".join(system_parts)

    # Build messages list
    messages = [{"role": "system", "content": system_prompt}]

    # Include recent conversation history (last 10 turns to save tokens)
    if req.history:
        for msg in req.history[-10:]:
            if msg.role in ("user", "assistant"):
                messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": req.message})

    try:
        response = client.chat.completions.create(
            model       = "gpt-4o-mini",
            messages    = messages,
            max_tokens  = 600,
            temperature = 0.7,
        )
        reply = response.choices[0].message.content.strip()
        return {"reply": reply}

    except openai.AuthenticationError:
        raise HTTPException(503, "OpenAI API key is invalid or expired. Check your OPENAI_API_KEY in .env.")
    except openai.RateLimitError as e:
        # openai-python raises RateLimitError (HTTP 429) for BOTH real rate
        # limiting AND "no credits left" quota errors. Inspect the error body
        # so we tell the user the true cause instead of a generic message.
        err_code = ""
        try:
            err_code = (e.body or {}).get("code", "") if isinstance(e.body, dict) else ""
        except Exception:
            pass
        if err_code == "credit_balance_exhausted" or "insufficient_quota" in str(e):
            raise HTTPException(
                429,
                "Your OpenAI account has no credits remaining. Add billing credits at "
                "https://platform.openai.com/settings/organization/billing, then try again."
            )
        raise HTTPException(429, "OpenAI rate limit reached. Wait a moment and try again.")
    except openai.APIConnectionError:
        raise HTTPException(503, "Cannot reach OpenAI servers. Check your internet connection.")
    except Exception as e:
        print(f"[ERROR] /chat OpenAI call failed: {e}")
        raise HTTPException(500, f"Chat failed: {type(e).__name__}: {e}")
