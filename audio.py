# Silence ffmpeg/OpenCV warnings before imports
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

"""
utils/audio.py
--------------
Audio feature extraction pipeline:
  1. Extract WAV from video using ffmpeg
  2. Load with librosa
  3. Compute MFCC (+ delta + delta-delta for richer features)
  4. Normalize per sample (zero-mean, unit-variance)
  5. Pad / truncate to fixed time length
  6. Return torch tensor  (1, n_mfcc, time_steps)

Changes vs original:
  • extract_wav_segment()   — extract only a [start_sec, end_sec] slice of
    audio from a video file.  Used by the timeline pipeline.
  • process_audio_segment() — full pipeline for a single time window:
    returns (tensor, energy) just like process_audio_file().
  • Uses FFMPEG constant from ffmpeg_utils for reliable subprocess calls.
"""

import os
import subprocess
import numpy as np
import librosa
import torch

from ffmpeg_utils import FFMPEG


# ── Config ───────────────────────────────────────────────────────────────────
N_MFCC      = 40     # number of MFCC coefficients
MAX_FRAMES  = 128    # fixed time-length (pad / truncate)
SAMPLE_RATE = 22050


def extract_wav_from_video(video_path: str, wav_path: str) -> bool:
    """
    Use ffmpeg to strip audio from a video file.
    capture_output=True suppresses all NAL/mmco warning spam from broken H264.
    """
    cmd = [
        FFMPEG, "-y",
        "-loglevel", "error",      # suppress NAL/mmco warnings
        "-i", video_path,
        "-vn",                     # no video
        "-acodec", "pcm_s16le",    # 16-bit PCM
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",                # mono
        wav_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def extract_wav_segment(video_path: str, wav_path: str,
                        start_sec: float, end_sec: float) -> bool:
    """
    Like extract_wav_from_video but only extracts the [start_sec, end_sec]
    slice of audio.  Uses ffmpeg's -ss (fast seek) and -t (duration).
    """
    duration = max(end_sec - start_sec, 0.5)
    cmd = [
        FFMPEG, "-y",
        "-loglevel", "error",
        "-ss", f"{start_sec:.3f}",   # fast seek before -i
        "-i", video_path,
        "-t", f"{duration:.3f}",     # how long to capture
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        wav_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def extract_mfcc(wav_path: str,
                 n_mfcc: int = N_MFCC,
                 max_frames: int = MAX_FRAMES) -> np.ndarray | None:
    """
    Load WAV → compute MFCC → pad/truncate → normalize → return (n_mfcc, T).
    Returns None if the file cannot be loaded.
    """
    try:
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  [audio] Cannot load {wav_path}: {e}")
        return None

    if len(y) < sr * 0.1:          # less than 100 ms → skip
        return None

    # 1. Compute MFCC  (shape: n_mfcc × T)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    features = mfcc   # shape: (n_mfcc, T)

    # 2. Pad or truncate time axis
    T = features.shape[1]
    if T < max_frames:
        pad = max_frames - T
        features = np.pad(features, ((0, 0), (0, pad)), mode="constant")
    else:
        features = features[:, :max_frames]

    # 3. Per-sample z-score normalization  ← FIX for same-prediction bug
    mean = features.mean()
    std  = features.std() + 1e-8
    features = (features - mean) / std

    return features.astype(np.float32)   # (n_mfcc, max_frames)


def mfcc_to_tensor(features: np.ndarray) -> torch.Tensor:
    """Convert (n_mfcc, T) numpy array to (1, n_mfcc, T) torch tensor."""
    return torch.tensor(features).unsqueeze(0)   # add channel dim


def audio_energy(wav_path: str) -> float:
    """Return RMS energy of audio — used for energy-based correction."""
    try:
        y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        return float(np.sqrt(np.mean(y ** 2)))
    except Exception:
        return 0.0


def process_audio_file(video_path: str,
                       tmp_dir: str = "/tmp") -> tuple[torch.Tensor | None, float]:
    """
    Full pipeline: video_path → wav → mfcc tensor + energy.
    Returns (tensor, energy) or (None, 0.0) on failure.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    wav_path = os.path.join(tmp_dir, "audio_tmp.wav")

    ok = extract_wav_from_video(video_path, wav_path)
    if not ok:
        print("  [audio] ffmpeg extraction failed")
        return None, 0.0

    feats = extract_mfcc(wav_path)
    if feats is None:
        return None, 0.0

    energy = audio_energy(wav_path)
    tensor = mfcc_to_tensor(feats)
    return tensor, energy


def process_audio_segment(video_path: str, start_sec: float, end_sec: float,
                           tmp_dir: str = "/tmp") -> tuple[torch.Tensor | None, float]:
    """
    Like process_audio_file but only processes the [start_sec, end_sec] window.
    Used by the emotion-timeline pipeline.
    Returns (tensor, energy) or (None, 0.0) on failure.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    wav_path = os.path.join(tmp_dir, f"seg_{int(start_sec)}_{int(end_sec)}.wav")

    ok = extract_wav_segment(video_path, wav_path, start_sec, end_sec)
    if not ok:
        return None, 0.0

    feats = extract_mfcc(wav_path)
    if feats is None:
        return None, 0.0

    energy = audio_energy(wav_path)
    tensor = mfcc_to_tensor(feats)
    return tensor, energy
