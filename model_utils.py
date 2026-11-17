"""
model_utils.py
Shared helpers for the Assignment 5 dashboard.

The audio model and feature extraction come from Assignment 4 (MFCC + delta
features -> small PyTorch network). They live in one place so that
train_models.py and streamlit_app.py always use *identical* preprocessing.
"""
import numpy as np
import torch.nn as nn
from scipy.io import wavfile
from python_speech_features import mfcc, delta

NUM_CEPS = 13  # 13 MFCC + 13 delta = 26 features (Assignment 4)
FEATURE_NAMES = [f"MFCC {i}" for i in range(1, NUM_CEPS + 1)] + \
                [f"Delta {i}" for i in range(1, NUM_CEPS + 1)]


class AudioSentimentNet(nn.Module):
    """Same architecture as Assignment 4."""

    def __init__(self, input_dim=26, hidden_dim=64, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_audio(source):
    """Read a WAV file (path or file-like object). Returns (sample_rate, mono float32 signal)."""
    sample_rate, signal = wavfile.read(source)
    signal = signal.astype(np.float32)
    if signal.ndim > 1:  # stereo -> mono
        signal = signal.mean(axis=1)
    return sample_rate, signal


def mfcc_delta_matrix(signal, sample_rate, num_ceps=NUM_CEPS):
    """Return (mfcc_matrix, combined_matrix) with shapes (T, 13) and (T, 26)."""
    # FFT size = next power of two >= one 25 ms window (512 for 16 kHz audio)
    nfft = 1 << int(np.ceil(np.log2(0.025 * sample_rate)))
    mfcc_feat = mfcc(signal, sample_rate, numcep=num_ceps, nfft=nfft)
    delta_feat = delta(mfcc_feat, 2)
    return mfcc_feat, np.hstack([mfcc_feat, delta_feat])


def extract_features(signal, sample_rate, num_ceps=NUM_CEPS):
    """Mean MFCC + delta vector across time -> fixed-length vector of 26 numbers."""
    _, combined = mfcc_delta_matrix(signal, sample_rate, num_ceps)
    return combined.mean(axis=0)
