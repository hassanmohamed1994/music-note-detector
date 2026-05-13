"""
EE410 — Training Script (PyTorch version)
==========================================
Run this once with your real .wav files to train and save the model.

SETUP: Organise your .wav files like this:
  data/
    C4/   your_c4_note1.wav   your_c4_note2.wav  ...
    E4/   your_e4_note1.wav   ...
    G4/   your_g4_note1.wav   ...

Then run:
  python train_model.py

Trained model is saved to: note_detector.pt
"""

from note_detector_nn import train

if __name__ == "__main__":
    train(
        data_dir="data",
        model_save_path="note_detector.pt",
        epochs=150,
        val_split=0.2,
    )
    print("\nDone! Model saved to note_detector.pt")
    print("Now run:  python note_detector_nn.py detect <your_file.wav>")
