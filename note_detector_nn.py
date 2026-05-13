"""
EE410 - Semester Project: Neural Network Musical Note Detector (PyTorch)
=========================================================================
Replaces threshold-based logic with a trained neural network.
Pipeline: WAV file → FIR filter bank (DSP front-end) → NN classifier (back-end)

Notes detected: C4 (261.63 Hz), E4 (329.63 Hz), G4 (392.00 Hz)

INSTALL:
    pip install torch scikit-learn scipy numpy
"""

import numpy as np
import os
import pickle

from scipy.signal import firwin, lfilter
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ─────────────────────────────────────────────────────────
#  SECTION 1 — DSP FRONT-END  (mirrors your MATLAB code)
# ─────────────────────────────────────────────────────────

# Filter bank: (label, center_hz, f_low, f_high)
FILTER_BANK = [
    ("C4", 261.63, 260.0, 263.0),
    ("E4", 329.63, 328.0, 331.0),
    ("G4", 392.00, 390.5, 393.5),
]

NOTE_LABELS = ["C4", "E4", "G4"]


def design_fir_filters(fs: int, order: int = 1023) -> list:
    """
    Design one bandpass FIR filter per note using firwin (= MATLAB fir1).
    Mirrors:  b = fir1(100, [f1 f2]/(fs/2))
    """
    nyq = fs / 2.0
    return [
        firwin(order + 1, [f_low / nyq, f_high / nyq], pass_zero=False)
        for _, _, f_low, f_high in FILTER_BANK
    ]


def extract_features(audio_path: str, filters=None) -> np.ndarray:
    """
    DSP front-end: WAV → mono → filter bank → NORMALISED ratio feature vector.

    Key design: instead of raw log-powers, we return the RATIO of each band's
    power to the total power across all bands. This makes features
    amplitude-independent — a quiet C4 and a loud C4 produce the same ratios.

    Returns shape (3,):  [ratio_C4, ratio_E4, ratio_G4]  (sums to 1.0)

    Mirrors your MATLAB:
        yE4     = filter(bE4, 1, x)
        powerE4 = sum(yE4.^2)
    """
    fs, x = wavfile.read(audio_path)

    # Normalise to float32 in [-1, 1]
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = x.astype(np.float32)

    # Stereo → mono (first channel, like your MATLAB code)
    if x.ndim > 1:
        x = x[:, 0]

    if filters is None:
        filters = design_fir_filters(fs)

    # Band power per filter  (sum of squared output samples)
    powers = np.array([
        float(np.sum(lfilter(b, [1.0], x) ** 2))
        for b in filters
    ], dtype=np.float32)

    # ── Amplitude-independent features ──────────────────────────────────────
    # Divide each band's power by the total power → ratio vector sums to 1.0
    # This means a quiet C4 and a loud C4 look identical to the network,
    # which is exactly what we want — we care about WHICH note, not how loud.
    total_power = powers.sum()
    if total_power < 1e-10:
        # Silent file — return uniform ratios
        return np.ones(len(FILTER_BANK), dtype=np.float32) / len(FILTER_BANK)

    ratios = powers / total_power
    return ratios


# ─────────────────────────────────────────────────────────
#  SECTION 2 — NEURAL NETWORK  (PyTorch)
# ─────────────────────────────────────────────────────────

class NoteDetectorNN(nn.Module):
    """
    Fully-connected classifier:
        Input(3) → Dense(32, ReLU) → Dropout(0.3)
                 → Dense(16, ReLU)
                 → Dense(3)   ← raw logits (CrossEntropyLoss handles softmax)

    3 inputs  = power ratio per filter band  [ratio_C4, ratio_E4, ratio_G4]
    3 outputs = score for each note [C4, E4, G4]
    """
    def __init__(self, input_dim: int = 3, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Return softmax probabilities as a numpy array."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1).numpy()


# ─────────────────────────────────────────────────────────
#  SECTION 3 — TRAINING
# ─────────────────────────────────────────────────────────

def prepare_dataset(data_dir: str):
    """
    Scan data_dir for WAV files in sub-folders named after each label:

        data/
          C4/  note_01.wav  note_02.wav ...
          E4/  note_01.wav ...
          G4/  note_01.wav ...

    Returns X (N×3 float32) and y (N,) int labels.
    """
    X, y = [], []

    for label_idx, label in enumerate(NOTE_LABELS):
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            print(f"  [WARNING] Folder not found: {folder} — skipping.")
            continue

        wav_files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
        if not wav_files:
            print(f"  [WARNING] No .wav files in {folder}")
            continue

        # Build filters once at this sample rate
        fs_sample, _ = wavfile.read(os.path.join(folder, wav_files[0]))
        filters = design_fir_filters(fs_sample)

        for fname in wav_files:
            fpath = os.path.join(folder, fname)
            try:
                X.append(extract_features(fpath, filters))
                y.append(label_idx)
            except Exception as e:
                print(f"  [ERROR] {fpath}: {e}")

    if not X:
        raise ValueError("No training data found. Check your data_dir structure.")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train(data_dir: str,
          model_save_path: str = "note_detector.pt",
          epochs: int = 150,
          batch_size: int = 16,
          val_split: float = 0.2,
          lr: float = 1e-3) -> NoteDetectorNN:
    """
    Full training loop.

    Args:
        data_dir        : folder with C4/, E4/, G4/ sub-folders of WAV files
        model_save_path : path to save the trained model (.pt)
        epochs          : max training epochs
        batch_size      : samples per gradient update
        val_split       : fraction of data held out for validation
        lr              : learning rate
    """
    print("=" * 55)
    print("  EE410 Neural Network Note Detector — Training")
    print("=" * 55)

    # ── Load data ──────────────────────────────────────────
    print("\n[1/4] Loading and processing audio files …")
    X, y = prepare_dataset(data_dir)
    print(f"      Dataset : {len(X)} samples | Classes: {NOTE_LABELS}")
    print(f"      Features: {X.shape}  (amplitude-independent band-power ratios)")
    print(f"      Sample feature vector: {X[0].round(4)}  (should sum to ~1.0: {X[0].sum():.4f})")

    # ── Normalise ──────────────────────────────────────────
    # StandardScaler still helps the NN converge faster even on ratio features
    print("\n[2/4] Normalising features (StandardScaler) …")
    scaler = StandardScaler()

    # Train / val split — skip validation if dataset is too small
    if len(X) < 15:
        print(f"\n      [WARNING] Only {len(X)} samples found.")
        print(f"      For best results add more WAV files (20+ per class recommended).")
        print(f"      Training on all samples, skipping validation split.\n")
        X_train, y_train = X, y
        X_val,   y_val   = X, y
    else:
        n_val = max(1, int(len(X) * val_split))
        idx = np.random.permutation(len(X))
        train_idx, val_idx = idx[n_val:], idx[:n_val]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]

    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True
    )

    # ── Build model ────────────────────────────────────────
    print("\n[3/4] Building model …")
    model = NoteDetectorNN()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n      Total trainable parameters: {total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    # ── Training loop ──────────────────────────────────────
    print("\n[4/4] Training …\n")
    best_val_acc     = 0.0
    best_state       = {k: v.clone() for k, v in model.state_dict().items()}
    patience_counter = 0
    EARLY_STOP_PATIENCE = 20

    X_val_t = torch.from_numpy(X_val_s)
    y_val_t = torch.from_numpy(y_val)

    for epoch in range(1, epochs + 1):
        # — Train —
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(yb)

        train_acc = correct / total

        # — Validate —
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss   = criterion(val_logits, y_val_t).item()
            val_acc    = (val_logits.argmax(1) == y_val_t).float().mean().item()

        scheduler.step(val_loss)

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} │ "
                  f"train loss: {total_loss/total:.4f}  acc: {train_acc*100:.1f}%  │ "
                  f"val loss: {val_loss:.4f}  acc: {val_acc*100:.1f}%")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    # Restore best weights
    model.load_state_dict(best_state)

    # ── Save ────────────────────────────────────────────────
    meta = {"scaler": scaler, "note_labels": NOTE_LABELS, "filter_bank": FILTER_BANK}
    torch.save({"model_state": model.state_dict(), "meta": meta}, model_save_path)
    print(f"\n  Model saved → {model_save_path}")
    print(f"\n{'─'*42}")
    print(f"  Best val accuracy : {best_val_acc*100:.1f}%")
    print(f"{'─'*42}\n")

    return model


# ─────────────────────────────────────────────────────────
#  SECTION 4 — INFERENCE  (replaces your MATLAB if/elseif block)
# ─────────────────────────────────────────────────────────

def load_model(model_path: str):
    """Load a saved model and its scaler/metadata."""
    checkpoint = torch.load(model_path, weights_only=False)
    meta  = checkpoint["meta"]
    model = NoteDetectorNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, meta["note_labels"], meta["scaler"]


def detect_note(filename: str, model_path: str = "note_detector.pt") -> str:
    """
    Main inference function — mirrors your MATLAB:
        function note = noteDetect(filename)

    Args:
        filename   : path to a .wav file
        model_path : path to the saved model (.pt)

    Returns:
        note : 'C4', 'E4', 'G4', or 'None'
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: '{model_path}'. Run train() first.")

    model, labels, scaler = load_model(model_path)

    # DSP front-end (identical preprocessing to training)
    ratios      = extract_features(filename)                        # (3,) ratio vector
    feat_scaled = scaler.transform(ratios[np.newaxis, :])          # normalise
    feat_tensor = torch.from_numpy(feat_scaled.astype(np.float32)) # → tensor

    # Neural network forward pass  ← replaces your if/elseif thresholds
    probs = model.predict_proba(feat_tensor)[0]   # shape (3,)

    confidence    = float(probs.max())
    predicted_idx = int(probs.argmax())

    CONFIDENCE_THRESHOLD = 0.60
    detected = labels[predicted_idx] if confidence >= CONFIDENCE_THRESHOLD else "None"

    # Pretty output
    print(f"\n{'='*44}")
    print(f"  File      : {os.path.basename(filename)}")
    print(f"  Ratios    : C4={ratios[0]:.4f}  E4={ratios[1]:.4f}  G4={ratios[2]:.4f}  (sum={ratios.sum():.4f})")
    print(f"  NN probs  : C4={probs[0]:.3f}  E4={probs[1]:.3f}  G4={probs[2]:.3f}")
    print(f"  Detected  : {detected}  (confidence {confidence*100:.1f}%)")
    print(f"{'='*44}\n")

    return detected


# ─────────────────────────────────────────────────────────
#  SECTION 5 — DEMO
# ─────────────────────────────────────────────────────────

def generate_synthetic_wav(freq_hz: float, out_path: str,
                            duration: float = 1.5, fs: int = 44100,
                            amplitude: float = 0.8,
                            noise_level: float = 0.02):
    """Generate a tone + harmonics + ADSR envelope .wav for testing."""
    dirpath = os.path.dirname(out_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    n = int(fs * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    signal = (
        1.00 * np.sin(2 * np.pi * 1 * freq_hz * t) +
        0.50 * np.sin(2 * np.pi * 2 * freq_hz * t) +
        0.25 * np.sin(2 * np.pi * 3 * freq_hz * t) +
        0.12 * np.sin(2 * np.pi * 4 * freq_hz * t)
    )
    signal /= np.max(np.abs(signal))

    # ADSR envelope
    a, d, r = int(0.02*fs), int(0.1*fs), int(0.2*fs)
    env = np.ones(n) * 0.7
    env[:a]    = np.linspace(0, 1, a)
    env[a:a+d] = np.linspace(1, 0.7, d)
    env[-r:]   = np.linspace(0.7, 0, r)
    signal *= env * amplitude

    signal += noise_level * np.random.randn(n)
    signal  = np.clip(signal, -1.0, 1.0)
    wavfile.write(out_path, fs, (signal * 32767).astype(np.int16))


def run_demo(num_per_class: int = 40):
    """End-to-end demo: generate data → train → test."""
    print("\n🎵  EE410 PyTorch Note Detector — Demo\n")

    note_freqs = {"C4": 261.63, "E4": 329.63, "G4": 392.00}

    print("[DEMO] Generating synthetic training data …")
    for label, freq in note_freqs.items():
        for i in range(num_per_class):
            generate_synthetic_wav(
                freq, f"demo_data/{label}/note_{i:02d}.wav",
                amplitude=np.random.uniform(0.3, 1.0),   # vary loudness
                noise_level=np.random.uniform(0.005, 0.03)
            )

    train("demo_data", model_save_path="demo_model.pt", epochs=150)

    print("[DEMO] Running inference on test files …\n")
    test_cases = [("C4", 261.63), ("E4", 329.63), ("G4", 392.00),
                  ("C4", 261.63), ("E4", 329.63), ("G4", 392.00)]
    correct = 0
    for i, (label, freq) in enumerate(test_cases):
        path = f"demo_test_{label}_{i}.wav"
        # Test with very different amplitude to prove amplitude-independence
        generate_synthetic_wav(freq, path,
                                amplitude=np.random.uniform(0.1, 0.5),
                                noise_level=0.02)
        result = detect_note(path, "demo_model.pt")
        status = "✓" if result == label else "✗"
        print(f"  {status}  Expected: {label}  →  Detected: {result}")
        correct += result == label

    print(f"\n  Demo accuracy: {correct}/{len(test_cases)}")

    # Cleanup
    import shutil
    for p in ["demo_data", "demo_model.pt"]:
        if os.path.exists(p):
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    for i, (label, _) in enumerate(test_cases):
        f = f"demo_test_{label}_{i}.wav"
        if os.path.exists(f):
            os.remove(f)


# ─────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        run_demo()
    elif sys.argv[1] == "train":
        data_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
        train(data_dir)
    elif sys.argv[1] == "detect":
        note = detect_note(sys.argv[2])
        print(f"Result: {note}")
    else:
        print("Usage:")
        print("  python note_detector_nn.py                    # run demo")
        print("  python note_detector_nn.py train <data_dir>   # train on your files")
        print("  python note_detector_nn.py detect <file.wav>  # detect a note")
