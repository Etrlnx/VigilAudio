import os
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter, convolve
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
DATA_ROOT = Path(r"D:\College stuff\Sem6\slp\slp_data")
MANIFEST_PATH = DATA_ROOT / "combined_manifest.csv"
SPOOFED_DIR = DATA_ROOT / "spoofed_data"
SPOOFED_MANIFEST_PATH = DATA_ROOT / "spoofed_manifest.csv"

TARGET_SR = 16000

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Creates a bandpass filter to simulate a cheap device speaker."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut=300.0, highcut=3400.0, fs=16000, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def generate_synthetic_rir(fs=16000, t50=0.3):
    """Generates a synthetic Room Impulse Response (RIR) with exponential decay."""
    t = np.arange(0, int(fs * t50)) / fs
    # Exponential decay to simulate reverberation in a standard room
    envelope = np.exp(-6 * np.log(10) * t / t50)
    noise = np.random.randn(len(t))
    rir = noise * envelope
    return rir / np.max(np.abs(rir)) # Normalize

def apply_replay_attack(audio, sr):
    """Simulates Speaker -> Room -> Microphone pipeline."""
    # 1. Speaker Simulation (Bandpass)
    audio_filtered = apply_bandpass(audio, fs=sr)
    
    # 2. Room Simulation (Reverb via Convolution)
    rir = generate_synthetic_rir(fs=sr, t50=0.25) # 250ms reverberation time
    audio_reverb = convolve(audio_filtered, rir, mode='full')[:len(audio)]
    
    # 3. Mic Noise Simulation (SNR ~ 30dB)
    noise_amp = 0.005 * np.random.randn(len(audio_reverb))
    audio_spoofed = audio_reverb + noise_amp
    
    # Normalize back to [-1.0, 1.0] to prevent audio clipping
    max_val = np.max(np.abs(audio_spoofed))
    if max_val > 0:
        audio_spoofed = audio_spoofed / max_val
        
    return audio_spoofed

def main():
    print("="*40)
    print("Generating Physical-Access Spoofed Database")
    print("="*40)
    
    df = pd.read_csv(MANIFEST_PATH)
    spoofed_records = []
    
    # Create parallel folder structure
    for split in ["train", "dev", "test"]:
        (SPOOFED_DIR / split).mkdir(parents=True, exist_ok=True)
        
    print(f"Processing {len(df)} files. This will take some time...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        orig_path = DATA_ROOT / row['path']
        split = row['split']
        filename = orig_path.name
        
        # New path for the spoofed file
        spoofed_rel_path = f"spoofed_data/{split}/{filename}"
        spoofed_abs_path = DATA_ROOT / spoofed_rel_path
        
        try:
            # 1. Load original pristine audio
            audio, _ = librosa.load(orig_path, sr=TARGET_SR, mono=True)
            
            # 2. Apply physical-access replay attack simulation
            spoofed_audio = apply_replay_attack(audio, TARGET_SR)
            
            # 3. Save the spoofed audio
            sf.write(spoofed_abs_path, spoofed_audio, TARGET_SR)
            
            # 4. Record in new manifest (adding the spoof flag)
            spoofed_records.append({
                "path": spoofed_rel_path,
                "language": row['language'],
                "label": row['label'],
                "split": split,
                "source": row['source'],
                "is_spoofed": 1  # 1 for spoofed, 0 for bona fide
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    # Save the parallel manifest
    spoofed_df = pd.DataFrame(spoofed_records)
    spoofed_df.to_csv(SPOOFED_MANIFEST_PATH, index=False)
    
    print("\n" + "="*40)
    print(f"SUCCESS! Created {len(spoofed_df)} spoofed audio files.")
    print(f"Manifest saved to: {SPOOFED_MANIFEST_PATH}")
    print("="*40)

if __name__ == "__main__":
    main()