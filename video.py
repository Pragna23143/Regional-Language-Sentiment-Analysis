import os
os.environ["OPENCV_LOG_LEVEL"]            = "SILENT"
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "1"

"""
utils/video.py  —  Speed-optimized + face-cropping
----------------------------------------------------
Changes vs original:
  • crop_face_from_frame()  — detect face with Haar cascade, crop to it,
    fall back to 80% centre-crop if no face found.
  • extract_frames_ffmpeg() now calls crop_face_from_frame() on every
    frame before adding it to the result list.
  • process_video_file_segment() — new helper that extracts frames only
    from a [start_sec, end_sec] window (used by the timeline endpoint).
  • Uses FFMPEG / FFPROBE constants from ffmpeg_utils for reliable subprocess calls.
"""

import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from ffmpeg_utils import FFMPEG, FFPROBE

FRAME_SIZE = 64
N_FRAMES   = 4        # reduced from 8 → 4 for 2× speed

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((FRAME_SIZE, FRAME_SIZE)),
    T.ToTensor(),
    NORMALIZE,
])

# ── Haar cascade (bundled with opencv-python, no extra downloads) ─────────────
_face_cascade = None

def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def crop_face_from_frame(bgr: np.ndarray) -> np.ndarray:
    """
    Detect the largest frontal face in *bgr* and return a tight crop of it
    (with 20 % padding on each side).  Falls back to an 80 % centre-crop of
    the original frame if no face is found.

    Always returns a BGR uint8 array.
    """
    h, w = bgr.shape[:2]
    cascade = _get_face_cascade()
    gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor  = 1.1,
        minNeighbors = 4,
        minSize      = (30, 30),
    )

    if len(faces) > 0:
        # Pick the largest detected face
        faces_sorted = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
        x, y, fw, fh = faces_sorted[0]

        # Add 20 % padding
        pad_x = int(fw * 0.20)
        pad_y = int(fh * 0.20)
        x1 = max(0,     x  - pad_x)
        y1 = max(0,     y  - pad_y)
        x2 = min(w,     x  + fw + pad_x)
        y2 = min(h,     y  + fh + pad_y)
        return bgr[y1:y2, x1:x2]

    # Fallback: 80 % centre-crop
    margin_x = int(w * 0.10)
    margin_y = int(h * 0.10)
    return bgr[margin_y:h - margin_y, margin_x:w - margin_x]


# ── Duration helper ───────────────────────────────────────────────────────────
def _get_duration(video_path: str) -> float:
    cmd = [
        FFPROBE, "-v", "quiet",
        "-print_format", "json",
        "-show_format", video_path
    ]
    try:
        import json
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=10)
        info = json.loads(result.stdout)
        return float(info.get("format", {}).get("duration", 10.0))
    except Exception:
        return 10.0


def _extract_one_frame(args):
    """Extract a single frame at a timestamp. Returns (index, bgr_array or None)."""
    video_path, i, ts, tmp_dir, frame_size = args
    out_png = os.path.join(tmp_dir, f"f{i:03d}.png")
    cmd = [
        FFMPEG, "-loglevel", "quiet",
        "-ss", f"{ts:.3f}",      # fast seek BEFORE -i
        "-i", video_path,
        "-frames:v", "1",
        "-vf", f"scale={frame_size*4}:{frame_size*4}:force_original_aspect_ratio=disable",
        "-q:v", "5",
        out_png, "-y",
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=10)
        if os.path.exists(out_png) and os.path.getsize(out_png) > 0:
            img = cv2.imread(out_png)
            if img is not None:
                # Apply face crop before returning
                img = crop_face_from_frame(img)
                return (i, img)
    except Exception:
        pass

    # Reliable fallback: use OpenCV's FFmpeg backend directly.  This is
    # especially useful on Windows when an FFmpeg subprocess cannot seek
    # correctly inside a downloaded H264/WebM file.
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)
            ok, img = cap.read()
            if ok and img is not None and img.size:
                img = crop_face_from_frame(img)
                return (i, img)
    except Exception:
        pass
    finally:
        if cap is not None:
            cap.release()

    return (i, None)


def extract_frames_ffmpeg(video_path: str, n_frames: int = N_FRAMES,
                          start_sec: float | None = None,
                          end_sec:   float | None = None) -> list:
    """
    Extract N frames in parallel using ThreadPoolExecutor.
    Optional start_sec / end_sec limit sampling to a time window (used by
    the timeline endpoint).
    """
    if not os.path.exists(video_path):
        return []

    duration   = max(_get_duration(video_path), 0.5)
    t_start    = start_sec if start_sec is not None else duration * 0.1
    t_end      = end_sec   if end_sec   is not None else duration * 0.9
    t_start    = max(0.0, min(t_start, duration))
    t_end      = max(t_start + 0.1, min(t_end, duration))

    tmp_dir    = tempfile.mkdtemp(prefix="eid_frames_")
    timestamps = np.linspace(t_start, t_end, n_frames)

    try:
        args_list = [(video_path, i, ts, tmp_dir, FRAME_SIZE)
                     for i, ts in enumerate(timestamps)]

        frames_dict = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_extract_one_frame, a): a for a in args_list}
            for future in as_completed(futures):
                idx, img = future.result()
                if img is not None:
                    frames_dict[idx] = img

        return [frames_dict[i] for i in sorted(frames_dict.keys())]

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def frames_to_tensor(frames: list):
    if not frames:
        return None
    tensors = []
    for bgr in frames:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensors.append(TRANSFORM(rgb))
    return torch.stack(tensors).mean(dim=0)


def process_video_file(video_path: str):
    """
    Extract frames and return a model-ready tensor.

    IMPORTANT: return None when no frame could be decoded.  Returning a
    zero-filled image makes the CNN appear to have succeeded and can hide
    real video-analysis failures.
    """
    frames = extract_frames_ffmpeg(video_path)
    return frames_to_tensor(frames)


def process_video_segment(video_path: str, start_sec: float, end_sec: float):
    """
    Like process_video_file but restricted to [start_sec, end_sec].
    Used by the emotion-timeline pipeline.
    """
    frames = extract_frames_ffmpeg(
        video_path, start_sec=start_sec, end_sec=end_sec
    )
    return frames_to_tensor(frames)
