"""
models/model.py
---------------
Two model architectures:
  1. AudioCNNLSTM  — CNN extracts local spectral patterns from MFCC,
                     LSTM captures temporal dynamics across time-steps.
  2. VideoCNN      — Lightweight CNN classifies a single video frame.

Both output logits over EMOTION_CLASSES (5 classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── shared constants ────────────────────────────────────────────────────────
EMOTION_CLASSES = ["angry", "fear", "happy", "neutral", "sad"]
NUM_EMOTIONS    = len(EMOTION_CLASSES)   # 5
INTENT_CLASSES  = ["aggressive", "conversational", "informative", "promotional"]
NUM_INTENTS     = len(INTENT_CLASSES)    # 4


# ── 1. Audio Model: CNN + LSTM ───────────────────────────────────────────────
class AudioCNNLSTM(nn.Module):
    """
    CNN encoder  →  LSTM sequence learner  →  FC classifier

    Input shape : (batch, 1, n_mfcc, time_steps)
                  e.g.  (B, 1, 40, 128)

    CNN role    : 2-D convolutions treat the MFCC spectrogram like an image.
                  Each filter learns a local pattern (e.g., voiced/unvoiced,
                  formant shape). Stacking two conv blocks doubles the
                  receptive field while halving the spatial size.

    LSTM role   : After CNN, we collapse the frequency axis and feed the
                  time-axis as a sequence.  The LSTM's hidden state remembers
                  context across the sequence, capturing how emotions evolve
                  over time (rising pitch, sustained anger, etc.).

    Dropout + BN: Prevent overfitting on a small dataset (~50 samples).
    """

    def __init__(self, n_mfcc: int = 40, lstm_hidden: int = 128,
                 lstm_layers: int = 2, num_classes: int = NUM_EMOTIONS,
                 dropout: float = 0.4):
        super().__init__()

        # ── CNN Encoder ──────────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (B,32,40,T)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),                          # (B,32,20,T/2)
            nn.Dropout2d(0.2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,64,20,T/2)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),                          # (B,64,10,T/2)
            nn.Dropout2d(0.2),
        )

        # After CNN the freq dim = n_mfcc // 4 = 10  (for n_mfcc=40)
        cnn_freq_out = n_mfcc // 4
        lstm_input   = 64 * cnn_freq_out   # flattened per time-step

        # ── LSTM Sequence Learner ────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = lstm_input,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            bidirectional = True,           # captures past + future context
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )

        # ── Classifier Head ──────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 64),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, n_mfcc, T)
        returns logits : (B, num_classes)
        """
        # CNN feature extraction
        feat = self.cnn(x)                  # (B, 64, freq', T')

        # Reshape for LSTM: treat time axis as sequence
        B, C, F, T = feat.shape
        feat = feat.permute(0, 3, 1, 2)     # (B, T', 64, freq')
        feat = feat.reshape(B, T, C * F)    # (B, T', lstm_input)

        # LSTM temporal learning
        lstm_out, _ = self.lstm(feat)       # (B, T', hidden*2)

        # Mean-pool across time then classify
        pooled = lstm_out.mean(dim=1)       # (B, hidden*2)
        logits = self.classifier(pooled)    # (B, num_classes)
        return logits


# ── 2. Video Model: Simple Visual CNN ────────────────────────────────────────
class VideoCNN(nn.Module):
    """
    Lightweight CNN for visual emotion detection from a single frame.

    Input  : (B, 3, 64, 64)  — RGB frame resized to 64×64
    Output : logits (B, num_classes)

    The CNN role here is to detect facial / scene cues:
    wrinkled brow → angry, wide eyes → fear, smile → happy, etc.
    """

    def __init__(self, num_classes: int = NUM_EMOTIONS, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),                  # 32x32
            nn.Dropout2d(0.2),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),                  # 16x16
            nn.Dropout2d(0.3),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),                  # 8x8
            nn.Dropout2d(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── 3. Fusion Helper ─────────────────────────────────────────────────────────
def fuse_predictions(audio_logits: torch.Tensor,
                     video_logits: torch.Tensor,
                     audio_weight: float = 0.65,
                     video_weight: float = 0.35) -> torch.Tensor:
    """
    Weighted probability fusion.
    Returns: combined probability tensor (num_classes,)
    """
    audio_probs = F.softmax(audio_logits, dim=-1)
    video_probs = F.softmax(video_logits, dim=-1)
    combined    = audio_weight * audio_probs + video_weight * video_probs
    return combined
