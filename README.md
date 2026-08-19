

# Audio Keyword Spotting & Classification with TensorFlow

An end-to-end Deep Learning pipeline that classifies spoken audio commands ("yes" vs "no") using Short-Time Fourier Transform (STFT) spectrograms and a 2D Convolutional Neural Network (CNN).

## Project Overview

- **Dataset:** TensorFlow Speech Commands Dataset (Mini version)
- **Task:** Binary Audio Classification (`yes` / `no`)
- **Signal Processing:** Raw audio waveforms (16 kHz mono) converted to 2D Spectrograms via STFT.
- **Model Architecture:** Custom 2D CNN with Normalization, Conv2D, MaxPool2D, Dropout, and Dense layers.
- **Test Accuracy:** ~92.5%

---

## Technical Pipeline

1. **Audio Preprocessing:**
   - Decodes 1-second WAV files into single-channel tensors at a 16 kHz sampling rate.
   - Converts time-domain waveforms into time-frequency domain **Spectrograms** using Short-Time Fourier Transform (`tf.signal.stft`).

2. **Model Architecture:**
   - `Resizing` layer (32x32) for fast spatial feature processing.
   - `Normalization` layer adapted to dataset spectrogram distribution.
   - Two `Conv2D` layers (32 & 64 filters) + `MaxPooling2D` + `Dropout` (0.25).
   - Fully Connected Dense layer (128 units) + `Dropout` (0.5) + Final Dense Output (2 logits).

3. **Training & Optimization:**
   - **Optimizer:** Adam
   - **Loss Function:** `SparseCategoricalCrossentropy(from_logits=True)`
   - **Regularization:** Early Stopping monitored on validation loss.

---

## Performance & Metrics

| Metric | Score |
| :--- | :--- |
| **Validation Accuracy** | 98.0% |
| **Test Accuracy** | 92.5% |
| **Test Loss** | 0.153 |

### Confusion Matrix
```text
Predicted ->   [No]  [Yes]
Actual [No]  [  84     3 ]
Actual [Yes] [  12   101 ]
