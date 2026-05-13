"""
EE410 — Training Data Generator (v2)
======================================
Generates realistic synthetic WAV files for C4, E4, and G4 notes.

Each file includes:
  - Fundamental frequency + harmonics (overtones 2x, 3x, 4x)
  - Natural ADSR amplitude envelope (attack/decay/sustain/release)
  - Randomised amplitude, duration, and noise per file
  - Slight pitch variation (+/- 0.5 Hz) to simulate real instrument drift

Run:
    python generate_data.py

Output:
    data/
      C4/  note_00.wav ... note_29.wav
      E4/  note_00.wav ... note_29.wav
      G4/  note_00.wav ... note_29.wav
"""

import numpy as np
import os
from scipy.io import wavfile

# ── Settings ──────────────────────────────────────────────────────────────────
FS            = 44100   # sample rate (Hz) — matches most real recordings
NUM_PER_CLASS = 30      # WAV files generated per note class
OUTPUT_DIR    = "data"  # top-level output folder

# Exact fundamental frequencies for each note
NOTES = {
    "C4": 261.63,
    "E4": 329.63,
    "G4": 392.00,
}

# ── ADSR Envelope ─────────────────────────────────────────────────────────────

def make_envelope(n_samples: int, fs: int,
                  attack_s:  float = 0.02,
                  decay_s:   float = 0.10,
                  sustain:   float = 0.75,
                  release_s: float = 0.25) -> np.ndarray:
    """
    ADSR amplitude envelope — makes tones sound like real instrument notes
    instead of abruptly switching on/off.

      Attack  (0.02s) : ramp 0 → 1
      Decay   (0.10s) : ramp 1 → sustain level
      Sustain         : steady at sustain level
      Release (0.25s) : ramp sustain → 0
    """
    env = np.ones(n_samples, dtype=np.float32)
    a = int(attack_s  * fs)
    d = int(decay_s   * fs)
    r = int(release_s * fs)

    env[:a]        = np.linspace(0.0, 1.0, a)          # attack
    env[a : a+d]   = np.linspace(1.0, sustain, d)      # decay
    env[a+d : -r]  = sustain                            # sustain
    env[-r:]       = np.linspace(sustain, 0.0, r)       # release
    return env


# ── Note Generator ────────────────────────────────────────────────────────────

def generate_note(freq:        float,
                  fs:          int   = FS,
                  duration:    float = 1.5,
                  amplitude:   float = 0.8,
                  noise_level: float = 0.015,
                  pitch_jitter: float = 0.3) -> np.ndarray:
    """
    Generate one synthetic note as a float32 array in [-1, 1].

    Parameters
    ----------
    freq         : fundamental frequency in Hz (e.g. 261.63 for C4)
    fs           : sample rate in Hz
    duration     : length of the note in seconds
    amplitude    : peak amplitude in [0, 1] — varies loudness between files
    noise_level  : amount of white noise added (simulates mic/room noise)
    pitch_jitter : max random pitch shift in Hz (simulates tuning drift)
    """
    n = int(fs * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    # Small random pitch shift so not every sample is exactly the same frequency
    f = freq + np.random.uniform(-pitch_jitter, pitch_jitter)

    # Fundamental + harmonics (like a real string/piano tone)
    # Each harmonic is half the amplitude of the previous one
    signal = (
        1.000 * np.sin(2 * np.pi * 1 * f * t) +   # fundamental  (100%)
        0.500 * np.sin(2 * np.pi * 2 * f * t) +   # 2nd harmonic  (50%)
        0.250 * np.sin(2 * np.pi * 3 * f * t) +   # 3rd harmonic  (25%)
        0.125 * np.sin(2 * np.pi * 4 * f * t) +   # 4th harmonic  (12.5%)
        0.062 * np.sin(2 * np.pi * 5 * f * t)     # 5th harmonic   (6%)
    )

    # Normalise to [-1, 1] before applying envelope and amplitude
    signal /= np.max(np.abs(signal))

    # Apply ADSR envelope with slight randomisation per note
    env = make_envelope(
        n, fs,
        attack_s  = np.random.uniform(0.01, 0.05),
        decay_s   = np.random.uniform(0.05, 0.15),
        sustain   = np.random.uniform(0.65, 0.85),
        release_s = np.random.uniform(0.15, 0.35),
    )
    signal *= env

    # Scale to desired amplitude (varies loudness between samples)
    signal *= amplitude

    # Add white noise (simulates microphone noise, room acoustics, etc.)
    signal += noise_level * np.random.randn(n).astype(np.float32)

    # Final safety clip
    signal = np.clip(signal, -1.0, 1.0)
    return signal.astype(np.float32)


# ── Save Utility ──────────────────────────────────────────────────────────────

def save_wav(signal: np.ndarray, path: str, fs: int = FS):
    """Save a float32 signal as a 16-bit PCM WAV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wavfile.write(path, fs, (signal * 32767).astype(np.int16))


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all(output_dir: str = OUTPUT_DIR,
                 num_per_class: int = NUM_PER_CLASS):

    print("=" * 54)
    print("  EE410 — Training Data Generator")
    print("=" * 54)
    print(f"\n  Notes      : {list(NOTES.keys())}")
    print(f"  Files/class: {num_per_class}  ({num_per_class * len(NOTES)} total)")
    print(f"  Sample rate: {FS} Hz")
    print(f"  Output dir : {output_dir}/\n")

    total = 0
    for label, freq in NOTES.items():
        print(f"  [{label}]  {freq} Hz …", end="  ")

        for i in range(num_per_class):
            signal = generate_note(
                freq,
                duration    = np.random.uniform(1.0, 2.0),    # 1–2 sec
                amplitude   = np.random.uniform(0.3, 0.95),   # vary loudness
                noise_level = np.random.uniform(0.005, 0.025),
                pitch_jitter= 0.3,                             # +/- 0.3 Hz drift
            )
            path = os.path.join(output_dir, label, f"note_{i:02d}.wav")
            save_wav(signal, path)
            total += 1

        print(f"saved {num_per_class} files → {output_dir}/{label}/")

    print(f"\n  ✓  Done! {total} WAV files written to '{output_dir}/'")
    print(f"\n  Next steps:")
    print(f"    1.  python train_model.py")
    print(f"    2.  python note_detector_nn.py detect <your_file.wav>\n")


if __name__ == "__main__":
    generate_all()
