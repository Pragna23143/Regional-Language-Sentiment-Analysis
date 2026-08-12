# Silence OpenCV H264 warnings BEFORE any import
import os
os.environ["OPENCV_LOG_LEVEL"]            = "SILENT"
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "1"

"""
train.py  —  adapted for actual dataset structure
==================================================

Your folder layout (detected automatically):
  data/raw/English/angry1/angryenglish1.mp4
  data/raw/English/criseseng1/crises1eng.mp4   <- mapped to "fear"
  data/raw/English/Happy1/happyeng1.mp4
  data/raw/English/neutral1/neutral1eng.mp4
  data/raw/English/sad1/sad1.mp4

  data/raw/kannada/Angrysample1/video.mp4
  data/raw/kannada/fear1/fear,crises.mp4
  data/raw/kannada/Happy1/happiness1.mp4
  data/raw/kannada/neutral1/neutral.mp4
  data/raw/kannada/Sadnesssample1/sadness1.mp4

Key differences from generic version
--------------------------------------
1. Emotion label detected from FOLDER NAME PREFIX, not folder name itself.
   e.g. "angry1" -> "angry",  "Angrysample4" -> "angry",
        "criseseng2" -> "fear", "Sadnesssample3" -> "sad"
2. WAV files already exist in each sample folder — used directly.
3. Language folders: "English" (capital E) and "kannada" (lowercase k).
4. "crises" prefix mapped to "fear" (same emotion, different name used).

Run:
    python train.py
"""

import os
import json
import glob
import random
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from model import AudioCNNLSTM, VideoCNN, EMOTION_CLASSES
from audio import extract_mfcc, N_MFCC, MAX_FRAMES
from video import process_video_file


DATA_ROOT = r"C:\Users\Pragna\Downloads\Emotion_Reco\Emotion_Recognition\data\raw"

CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE     = 8
EPOCHS         = 20
LR             = 1e-3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED           = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


PREFIX_TO_EMOTION = {
    "angry":     "angry",
    "crises":    "fear",      # English "crises" folders = fear emotion
    "fear":      "fear",      # Kannada fear folders
    "happy":     "happy",
    "happiness": "happy",     # Kannada Happy folders have "happiness" files
    "neutral":   "neutral",
    "sad":       "sad",
    "sadness":   "sad",       # Kannada Sadnesssample folders
}

EMOTION_LABEL_MAP = {em: i for i, em in enumerate(sorted(EMOTION_CLASSES))}



def folder_to_emotion(folder_name: str):
    """
    Derive emotion label from a sample subfolder name.
    Examples:
      "angry1"        -> "angry"
      "Angrysample4"  -> "angry"
      "criseseng2"    -> "fear"
      "fear3"         -> "fear"
      "Happy6"        -> "happy"
      "neutral2"      -> "neutral"
      "sad5"          -> "sad"
      "Sadnesssample3"-> "sad"
    """
    name_lower = folder_name.lower()
    for prefix, emotion in PREFIX_TO_EMOTION.items():
        if name_lower.startswith(prefix):
            return emotion
    return None


# ── Dataset scanner ───────────────────────────────────────────────────────────
def collect_samples():
    """
    Walk English/ and kannada/ subfolders.
    For every sample subfolder, find the .mp4 and its .wav/.mp3 audio.
    Returns list of dicts: {video, wav, label, language, emotion, folder}
    """
    samples = []

    # Your actual language folder names (case-sensitive on Windows too)
    lang_dirs = {
        "english": os.path.join(DATA_ROOT, "English"),
        "kannada": os.path.join(DATA_ROOT, "kannada"),
    }

    for lang_key, lang_dir in lang_dirs.items():
        if not os.path.isdir(lang_dir):
            print(f"  [warn] Not found: {lang_dir}")
            continue

        for sample_folder in sorted(os.listdir(lang_dir)):
            sample_path = os.path.join(lang_dir, sample_folder)
            if not os.path.isdir(sample_path):
                continue

            emotion = folder_to_emotion(sample_folder)
            if emotion is None:
                print(f"  [skip] Cannot map '{sample_folder}' to emotion")
                continue

            label = EMOTION_LABEL_MAP[emotion]

            # Find video file
            mp4s = glob.glob(os.path.join(sample_path, "*.mp4"))
            if not mp4s:
                print(f"  [skip] No .mp4 in {sample_path}")
                continue
            video_path = mp4s[0]

            # Find audio file (prefer wav, accept mp3)
            wavs = (glob.glob(os.path.join(sample_path, "*.wav")) +
                    glob.glob(os.path.join(sample_path, "*.mp3")))
            wav_path = wavs[0] if wavs else None

            samples.append({
                "video":    video_path,
                "wav":      wav_path,
                "label":    label,
                "language": lang_key,
                "emotion":  emotion,
                "folder":   sample_folder,
            })

    # Print summary
    print(f"\n[dataset] Found {len(samples)} samples total\n")
    for em, idx in sorted(EMOTION_LABEL_MAP.items(), key=lambda x: x[1]):
        eng = sum(1 for s in samples if s["label"] == idx and s["language"] == "english")
        kan = sum(1 for s in samples if s["label"] == idx and s["language"] == "kannada")
        print(f"  {em:8s} (label={idx}) : English={eng}  Kannada={kan}")
    print()
    return samples


# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item  = self.samples[idx]
        label = item["label"]

        # ── Audio features ────────────────────────────────────────────────
        audio_t = torch.zeros(1, N_MFCC, MAX_FRAMES)
        wav_path = item.get("wav")

        if wav_path and os.path.exists(wav_path):
            # Use existing WAV/MP3 directly — already in your dataset!
            feats = extract_mfcc(wav_path)
            if feats is not None:
                audio_t = torch.tensor(feats).unsqueeze(0)
        else:
            # Fallback: extract WAV from MP4 on the fly
            tmp_wav = os.path.join(tempfile.gettempdir(),
                                   f"eid_{os.getpid()}_{idx}.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", item["video"], "-vn",
                 "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", tmp_wav],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            feats = extract_mfcc(tmp_wav)
            if feats is not None:
                audio_t = torch.tensor(feats).unsqueeze(0)

        # ── Video features ────────────────────────────────────────────────
        vid_t = process_video_file(item["video"])
        if vid_t is None:
            vid_t = torch.zeros(3, 64, 64)

        return audio_t, vid_t, label


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_class_weights(samples, num_classes):
    counts  = np.bincount([s["label"] for s in samples], minlength=num_classes)
    counts  = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def train_model(model, train_loader, val_loader, mode="audio", epochs=EPOCHS):
    model.to(DEVICE)
    cw        = compute_class_weights(train_loader.dataset.samples,
                                      len(EMOTION_CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0., 0, 0
        for audio_t, video_t, labels in train_loader:
            labels = labels.to(DEVICE)
            x      = audio_t.to(DEVICE) if mode == "audio" else video_t.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            t_loss    += loss.item() * labels.size(0)
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total   += labels.size(0)

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for audio_t, video_t, labels in val_loader:
                labels = labels.to(DEVICE)
                x      = audio_t.to(DEVICE) if mode == "audio" else video_t.to(DEVICE)
                preds  = model(x).argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total   += labels.size(0)

        val_acc = v_correct / max(v_total, 1) * 100
        scheduler.step(t_loss / t_total)
        print(f"  Epoch {epoch:02d}/{epochs} | "
              f"Loss {t_loss/t_total:.4f} | "
              f"Train {t_correct/t_total*100:.1f}% | "
              f"Val {val_acc:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, f"{mode}_model_best.pth"))

    torch.save(model.state_dict(),
               os.path.join(CHECKPOINT_DIR, f"{mode}_model.pth"))
    print(f"\n  [{mode}] Best val acc: {best_val_acc:.1f}%")
    print(f"  [{mode}] Saved -> checkpoints/{mode}_model.pth\n")



def main():
    print("=" * 60)
    print("  Emotion & Intent — Training Pipeline")
    print(f"  Device : {DEVICE}")
    print(f"  Data   : {DATA_ROOT}")
    print("=" * 60 + "\n")

    samples = collect_samples()
    if not samples:
        print("[ERROR] No samples found. Check DATA_ROOT path in train.py.")
        return

    
    with open(os.path.join(CHECKPOINT_DIR, "label_map.json"), "w") as f:
        json.dump(EMOTION_LABEL_MAP, f, indent=2)
    print(f"[info] Label map saved -> {CHECKPOINT_DIR}/label_map.json\n")

   
    labels_for_split = [s["label"] for s in samples]
    try:
        train_s, val_s = train_test_split(
            samples, test_size=0.2, random_state=SEED,
            stratify=labels_for_split)
    except ValueError:
        train_s, val_s = train_test_split(
            samples, test_size=0.2, random_state=SEED)

    print(f"[split] Train: {len(train_s)}  Val: {len(val_s)}\n")

    
    sample_labels  = [s["label"] for s in train_s]
    class_counts   = np.bincount(sample_labels, minlength=len(EMOTION_CLASSES))
    class_counts   = np.where(class_counts == 0, 1, class_counts)
    sample_weights = [1.0 / class_counts[l] for l in sample_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(EmotionDataset(train_s), batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0)
    val_loader   = DataLoader(EmotionDataset(val_s),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print("─" * 60)
    print("  Training Audio CNN+LSTM")
    print("─" * 60)
    audio_model = AudioCNNLSTM(n_mfcc=N_MFCC, num_classes=len(EMOTION_CLASSES))
    train_model(audio_model, train_loader, val_loader, mode="audio")

    print("─" * 60)
    print("  Training Video CNN")
    print("─" * 60)
    video_model = VideoCNN(num_classes=len(EMOTION_CLASSES))
    train_model(video_model, train_loader, val_loader, mode="video")

    print("=" * 60)
    print("  Training complete!")
    print("  Next: python test.py --url <youtube_link> --language english")
    print("=" * 60)


if __name__ == "__main__":
    main()
