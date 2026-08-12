"""
test.py  —  Inference pipeline
================================
Changes vs original:
  • predict_timeline() — new function that processes the whole video in
    overlapping windows (default 8 s each, 4 s stride) and returns a list of
    { timestamp, audio_emotion, fused_emotion, audio_probs, fused_probs }.
  • predict() unchanged except for an import of the new segment helpers.
  • Uses FFPROBE constant from ffmpeg_utils for reliable subprocess calls.

Run:
    python test.py --url "https://youtu.be/xxxx" --language english
    python test.py --file "C:/path/to/video.mp4" --language english
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ── Auto-locate ffmpeg/ffprobe before anything else ───────────────────────────
import ffmpeg_utils  # noqa: F401  (side-effect: sets PATH)
from ffmpeg_utils import FFPROBE

os.environ["OPENCV_LOG_LEVEL"]            = "SILENT"
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "1"

import json
import argparse
import tempfile
import subprocess

import cv2
import torch
import torch.nn.functional as F

from model import AudioCNNLSTM, VideoCNN, EMOTION_CLASSES, fuse_predictions
from audio import (process_audio_file, extract_wav_from_video,
                   process_audio_segment, N_MFCC, SAMPLE_RATE)
from video import process_video_file, process_video_segment
from youtube import download_youtube_video, transcribe_audio
from intent import detect_intent

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    audio_model = AudioCNNLSTM(n_mfcc=N_MFCC, num_classes=len(EMOTION_CLASSES))
    video_model = VideoCNN(num_classes=len(EMOTION_CLASSES))

    audio_path = os.path.join(CHECKPOINT_DIR, "audio_model.pth")
    video_path = os.path.join(CHECKPOINT_DIR, "video_model.pth")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio model not found at {audio_path}. Run train.py first.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video model not found at {video_path}. Run train.py first.")

    audio_model.load_state_dict(
        torch.load(audio_path, map_location=DEVICE, weights_only=True))
    video_model.load_state_dict(
        torch.load(video_path, map_location=DEVICE, weights_only=True))
    audio_model.eval().to(DEVICE)
    video_model.eval().to(DEVICE)
    return audio_model, video_model


def _get_video_duration(video_path: str) -> float:
    """Return duration via ffprobe, with an OpenCV fallback."""
    cmd = [
        FFPROBE, "-v", "quiet",
        "-print_format", "json",
        "-show_format", video_path,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=15)
        info = json.loads(result.stdout)
        duration = float(info.get("format", {}).get("duration", 0.0))
        if duration > 0:
            return duration
    except Exception:
        pass

    # Some Windows FFmpeg builds fail to probe certain downloaded containers.
    # OpenCV can still read their frame count/FPS in many of those cases.
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            cap.release()
            if fps > 0 and frames > 0:
                return frames / fps
        else:
            cap.release()
    except Exception:
        pass

    return 0.0


def predict(video_path, language="english",
            audio_model=None, video_model=None):

    tmp_dir = tempfile.mkdtemp(prefix="eid_infer_")

    # ── wav path MUST match what process_audio_file creates ───────────────────
    wav_path = os.path.join(tmp_dir, "audio_tmp.wav")

    # ── 1. Audio features ─────────────────────────────────────────────────────
    print("[infer] Extracting audio features...")
    audio_tensor, energy = process_audio_file(video_path, tmp_dir)

    audio_emotion    = "neutral"
    audio_probs_dict = {e: round(1.0 / len(EMOTION_CLASSES), 4) for e in EMOTION_CLASSES}

    if audio_tensor is not None and audio_model is not None:
        with torch.no_grad():
            logits = audio_model(audio_tensor.unsqueeze(0).to(DEVICE))
            probs  = F.softmax(logits, dim=-1)[0].cpu()

        top2_vals, top2_idx = probs.topk(2)
        if (top2_vals[0] - top2_vals[1]) < 0.08:
            predicted_idx = top2_idx[1].item()
        else:
            predicted_idx = top2_idx[0].item()

        if energy < 0.01 and EMOTION_CLASSES[predicted_idx] == "angry":
            predicted_idx = EMOTION_CLASSES.index("neutral")

        audio_emotion    = EMOTION_CLASSES[predicted_idx]
        audio_probs_dict = {EMOTION_CLASSES[i]: round(probs[i].item(), 4)
                            for i in range(len(EMOTION_CLASSES))}

    print(f"  Audio emotion : {audio_emotion}")
    print(f"  Audio probs   : {audio_probs_dict}")

    # ── 2. Video features ─────────────────────────────────────────────────────
    print("[infer] Extracting video features...")
    video_tensor = process_video_file(video_path)

    video_available = video_tensor is not None
    video_emotion    = "neutral"
    video_probs_dict = {e: round(1.0 / len(EMOTION_CLASSES), 4) for e in EMOTION_CLASSES}

    if video_tensor is not None and video_model is not None:
        with torch.no_grad():
            logits = video_model(video_tensor.unsqueeze(0).to(DEVICE))
            probs  = F.softmax(logits, dim=-1)[0].cpu()

        predicted_idx    = probs.argmax().item()
        video_emotion    = EMOTION_CLASSES[predicted_idx]
        video_probs_dict = {EMOTION_CLASSES[i]: round(probs[i].item(), 4)
                            for i in range(len(EMOTION_CLASSES))}

    print(f"  Video emotion : {video_emotion}")

    # ── 3. Fusion ──────────────────────────────────────────────────────────────
    if audio_tensor is not None and video_tensor is not None:
        a_t = torch.tensor([audio_probs_dict[e] for e in EMOTION_CLASSES])
        v_t = torch.tensor([video_probs_dict[e] for e in EMOTION_CLASSES])
        fused = fuse_predictions(a_t.unsqueeze(0), v_t.unsqueeze(0))
        final_emotion = EMOTION_CLASSES[fused.argmax().item()]
    elif audio_tensor is not None:
        final_emotion = audio_emotion
    elif video_tensor is not None:
        final_emotion = video_emotion
    else:
        final_emotion = "neutral"

    print(f"  Fused emotion : {final_emotion}")

    # ── 4. Transcription ───────────────────────────────────────────────────────
    print("[infer] Transcribing audio...")
    transcript = ""

    # If wav wasn't created by process_audio_file, try extracting again
    if not os.path.exists(wav_path):
        print("  [transcript] WAV not found, extracting again...")
        extract_wav_from_video(video_path, wav_path)

    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
        try:
            print(f"  [transcript] WAV found ({os.path.getsize(wav_path)} bytes), running Whisper...")
            transcript = transcribe_audio(wav_path, language=language)
            print(f"  [transcript] Done: {transcript[:100]}")
        except Exception as e:
            print(f"  [warn] Transcription failed: {e}")
            transcript = "(transcription unavailable)"
    else:
        print("  [warn] WAV file missing or empty — skipping transcription")
        transcript = "(audio extraction failed)"

    # ── 5. Intent detection ────────────────────────────────────────────────────
    print("[infer] Detecting intent...")
    intent, intent_probs, intent_explanation = detect_intent(
        transcript, emotion=final_emotion)
    print(f"  Intent        : {intent}")

    return {
        "emotion"            : final_emotion,
        "intent"             : intent,
        "intent_explanation" : intent_explanation,
        "language"           : language,
        "transcript"         : transcript,
        "audio_emotion"      : audio_emotion,
        "video_emotion"      : video_emotion,
        "audio_probs"        : audio_probs_dict,
        "video_probs"        : video_probs_dict,
        "intent_probs"       : {k: round(v, 4) for k, v in intent_probs.items()},
        "audio_energy"       : round(energy, 6),
        "audio_available"    : audio_tensor is not None,
        "video_available"    : video_available,
    }


def predict_timeline(video_path: str,
                     language:    str  = "english",
                     audio_model        = None,
                     video_model        = None,
                     window_secs: float = 8.0,
                     stride_secs: float = 8.0) -> list[dict]:
    """
    Process the entire video in non-overlapping windows of *window_secs* and
    return a timeline list.

    Each entry is::

        {
          "timestamp"     : float,          # window start time in seconds
          "audio_emotion" : str,
          "fused_emotion" : str,
          "audio_probs"   : {str: float},
          "fused_probs"   : {str: float},
        }

    Uses process_audio_segment() and process_video_segment() so no full-video
    re-download is needed.
    """
    duration = _get_video_duration(video_path)
    if duration <= 0:
        raise RuntimeError(
            "Could not determine video duration. The downloaded video could "
            "not be decoded by FFmpeg or OpenCV."
        )

    timeline = []
    t = 0.0
    win_idx = 0

    while t < duration:
        t_end = min(t + window_secs, duration)
        print(f"[timeline] Window {win_idx}: {t:.1f}s – {t_end:.1f}s")

        tmp_dir = tempfile.mkdtemp(prefix=f"eid_tl_{win_idx}_")

        try:
            # Reset per-window state so a failed window can never reuse the
            # previous window's video/audio result.
            audio_tensor = None
            energy = 0.0
            video_tensor = None
            video_available = False
            video_emotion = "neutral"
            video_probs_dict = {
                e: round(1.0 / len(EMOTION_CLASSES), 4)
                for e in EMOTION_CLASSES
            }

            # ── Audio for this window ─────────────────────────────────────────
            audio_tensor, energy = process_audio_segment(
                video_path, t, t_end, tmp_dir)

            audio_emotion    = "neutral"
            audio_probs_dict = {e: round(1.0 / len(EMOTION_CLASSES), 4)
                                for e in EMOTION_CLASSES}

            if audio_tensor is not None and audio_model is not None:
                with torch.no_grad():
                    logits = audio_model(audio_tensor.unsqueeze(0).to(DEVICE))
                    probs  = F.softmax(logits, dim=-1)[0].cpu()

                top2_vals, top2_idx = probs.topk(2)
                if (top2_vals[0] - top2_vals[1]) < 0.08:
                    pidx = top2_idx[1].item()
                else:
                    pidx = top2_idx[0].item()
                if energy < 0.01 and EMOTION_CLASSES[pidx] == "angry":
                    pidx = EMOTION_CLASSES.index("neutral")

                audio_emotion    = EMOTION_CLASSES[pidx]
                audio_probs_dict = {EMOTION_CLASSES[i]: round(probs[i].item(), 4)
                                    for i in range(len(EMOTION_CLASSES))}

            # ── Video for this window ─────────────────────────────────────────
            video_tensor = process_video_segment(video_path, t, t_end)
            video_available = video_tensor is not None

            video_emotion = "neutral"
            video_probs_dict = {e: round(1.0 / len(EMOTION_CLASSES), 4)
                                for e in EMOTION_CLASSES}

            if video_tensor is not None and video_model is not None:
                with torch.no_grad():
                    logits = video_model(video_tensor.unsqueeze(0).to(DEVICE))
                    probs = F.softmax(logits, dim=-1)[0].cpu()
                video_emotion = EMOTION_CLASSES[probs.argmax().item()]
                video_probs_dict = {
                    EMOTION_CLASSES[i]: round(probs[i].item(), 4)
                    for i in range(len(EMOTION_CLASSES))
                }

            # ── Fuse only the modalities that actually decoded ───────────────
            if audio_tensor is not None and video_tensor is not None:
                a_t = torch.tensor(
                    [audio_probs_dict[e] for e in EMOTION_CLASSES],
                    dtype=torch.float32
                )
                v_t = torch.tensor(
                    [video_probs_dict[e] for e in EMOTION_CLASSES],
                    dtype=torch.float32
                )
                fused_probs_t = fuse_predictions(
                    a_t.unsqueeze(0), v_t.unsqueeze(0)
                )[0]
            elif audio_tensor is not None:
                fused_probs_t = torch.tensor(
                    [audio_probs_dict[e] for e in EMOTION_CLASSES],
                    dtype=torch.float32
                )
            elif video_tensor is not None:
                fused_probs_t = torch.tensor(
                    [video_probs_dict[e] for e in EMOTION_CLASSES],
                    dtype=torch.float32
                )
            else:
                raise RuntimeError("Neither audio nor video could be decoded for this window.")

            fused_emotion = EMOTION_CLASSES[fused_probs_t.argmax().item()]
            fused_probs_dict = {
                EMOTION_CLASSES[i]: round(fused_probs_t[i].item(), 4)
                for i in range(len(EMOTION_CLASSES))
            }

        except Exception as exc:
            print(f"  [timeline] Window {win_idx} failed: {exc}")
            audio_emotion    = "neutral"
            fused_emotion    = "neutral"
            audio_probs_dict = {e: round(1.0 / len(EMOTION_CLASSES), 4) for e in EMOTION_CLASSES}
            fused_probs_dict = audio_probs_dict.copy()
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        timeline.append({
            "timestamp"        : round(t, 2),
            "audio_emotion"    : audio_emotion,
            "video_emotion"    : video_emotion,
            "fused_emotion"    : fused_emotion,
            "audio_probs"      : audio_probs_dict,
            "video_probs"      : video_probs_dict,
            "fused_probs"      : fused_probs_dict,
            "audio_available"  : audio_tensor is not None,
            "video_available"  : video_available,
        })

        t += stride_secs
        win_idx += 1

    print(f"[timeline] Done — {len(timeline)} windows processed.")
    return timeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",      type=str, help="YouTube URL")
    parser.add_argument("--file",     type=str, help="Local video file path")
    parser.add_argument("--language", type=str, default="english",
                        choices=["english", "kannada"])
    args = parser.parse_args()

    if not args.url and not args.file:
        parser.error("Provide --url or --file")

    print("\n" + "=" * 60)
    print("  Emotion & Intent Detector — Inference")
    print("=" * 60 + "\n")

    print("[init] Loading models...")
    audio_model, video_model = load_models()

    if args.url:
        print(f"[init] Downloading: {args.url}")
        video_path = download_youtube_video(args.url, output_dir=tempfile.mkdtemp())
        if not video_path:
            print("[ERROR] Download failed.")
            return
    else:
        video_path = args.file
        if not os.path.exists(video_path):
            print(f"[ERROR] File not found: {video_path}")
            return

    print(f"[init] Video: {video_path}\n")
    result = predict(video_path, args.language, audio_model, video_model)

    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
