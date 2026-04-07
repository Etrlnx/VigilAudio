# SentinL-Indic: Multi-Task XLS-R for Language ID & Anti-Spoofing

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-latest-orange.svg)

**SentinL-Indic** is a robust speech processing framework designed to identify regional Indic languages while simultaneously verifying audio authenticity. Built on the **Wav2Vec2-XLS-R-53** backbone, this model implements a "rejection-first" architecture to filter out out-of-distribution (OOD) "spam" and synthetic spoofing attacks.

---

## Key Features
- **Multi-Task Learning (MTL):** Shared encoder with dual-head classification for Language ID and Spoof Detection.
- **Regional Focus:** Optimized for **Bengali (BN)**, **Nepali (NE)**, and **Assamese (AS)**.
- **OOD Rejection:** A dedicated 4th class ("Spam") trained on English and Hindi to prevent false positives in production.
- **Security-First:** Integrated **Equal Error Rate (EER)** calculation for biometric liveness verification.
- **Efficiency:** Uses **Automatic Mixed Precision (AMP)** and Gradient Clipping for stable training on consumer GPUs (e.g., RTX 30/40/50 series).

---

## Tech Stack
- **Backbone:** `facebook/wav2vec2-large-xlsr-53`
- **Audio Processing:** Librosa, Torchaudio
- **Deep Learning:** PyTorch, Hugging Face Transformers
- **Metrics:** Scikit-Learn (Macro F1, Confusion Matrices), SciPy (EER)
