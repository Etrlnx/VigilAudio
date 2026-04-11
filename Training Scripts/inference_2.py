import os
import argparse
import torch
import librosa
import numpy as np
from tqdm import tqdm
from model_2 import initialize_model

# --- Configuration ---
LANG_NAMES = ["Bengali", "Nepali", "Assamese", "Spam/OOD"]
SPOOF_NAMES = ["Bona Fide (Safe)", "Spoof Detected (Threat)"]

def load_trained_model(weights_path, device, num_langs=4):
    model = initialize_model(device=device, num_langs=num_langs)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file '{weights_path}' not found!")
        
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model Loaded Successfully from {weights_path}")
    return model

def run_inference_on_file(model, input_path, device):
    """Processes a single audio file and returns results."""
    # librosa.load supports .mp3 and .wav automatically if ffmpeg is installed
    try:
        y, sr = librosa.load(input_path, sr=16000, mono=True)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return None

    max_len = 16000 * 4
    
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))
    
    # Normalization
    if np.max(np.abs(y)) > 1e-6:
        y = y / np.max(np.abs(y))

    waveform = torch.from_numpy(y).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
            lang_logits, spoof_logits = model(waveform)
            
            lang_probs = torch.softmax(lang_logits, dim=1)
            spoof_probs = torch.softmax(spoof_logits, dim=1)

    lang_idx = torch.argmax(lang_probs).item()
    spoof_idx = torch.argmax(spoof_probs).item()
    
    return {
        "lang": LANG_NAMES[lang_idx],
        "lang_conf": lang_probs[0][lang_idx].item(),
        "security": SPOOF_NAMES[spoof_idx],
        "security_conf": spoof_probs[0][spoof_idx].item()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="inference_data", help="Folder containing audio files")
    parser.add_argument('--weights', type=str, default="best_multitask_model.pt")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = load_trained_model(args.weights, DEVICE)
        
        # --- CRITICAL CHANGE: Detect both .wav and .mp3 files ---
        # Added .mp3 to the tuple of allowed extensions
        valid_extensions = ('.wav', '.mp3')
        audio_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_extensions)]
        audio_files.sort() 
        
        if not audio_files:
            print(f"No compatible audio files found in {args.input_dir}")
        else:
            print(f"Found {len(audio_files)} files. Starting Batch Inference...\n")

            for filename in tqdm(audio_files, desc="Processing Audio", unit="file"):
                file_path = os.path.join(args.input_dir, filename)
                
                res = run_inference_on_file(model, file_path, DEVICE)
                
                if res:
                    print(f"\n[FILE]: {filename}")
                    print(f"   ├─ Language ID    : {res['lang']} ({res['lang_conf']:.1%})")
                    print(f"   └─ Security Status : {res['security']} ({res['security_conf']:.1%})")
                    print("-" * 50)

    except Exception as e:
        print(f"Error during inference: {e}")