import torch
import pandas as pd
import librosa
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# --- Configuration ---
DATA_ROOT = Path(r"D:\College stuff\Sem6\slp\slp_data")

class SLPDataset(Dataset):
    def __init__(self, manifest_file, split="train", max_seconds=4, target_sr=16000):
        """
        Custom Dataset for Multi-Task Learning.
        Returns: waveform, lang_label (0-3), spoof_label (0-1)
        """
        df = pd.read_csv(manifest_file)
        
        # --- CRITICAL FIX: Ensure Spoof column is integer and has no NaNs ---
        # If 'spoof' is empty (common in newly added spam data), default to 0 (Bona Fide)
        df['spoof'] = df['spoof'].fillna(0).astype(int)
        df['label'] = df['label'].astype(int)
        
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.max_length = max_seconds * target_sr
        self.target_sr = target_sr
        self.split = split
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Path and Labels
        audio_path = str(DATA_ROOT / row['path'])
        lang_label = row['label']
        spoof_label = row['spoof'] 
        
        # 2. Load Audio
        try:
            # We load with mono=True to ensure 1D shape [Samples]
            waveform_np, _ = librosa.load(audio_path, sr=self.target_sr, mono=True)
            waveform = torch.from_numpy(waveform_np)
        except Exception as e:
            # Fallback if a file is missing or corrupted
            waveform = torch.zeros(self.max_length)
            
        # 3. Standardize Length (Pad or Truncate)
        current_length = waveform.shape[0]
        if current_length > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            pad_amount = self.max_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        # 4. Normalization (Acoustic consistency)
        if torch.abs(waveform).max() > 1e-6:
            waveform = waveform / torch.abs(waveform).max()

        # Return waveform and both labels for MTL
        return (
            waveform, 
            torch.tensor(lang_label, dtype=torch.long), 
            torch.tensor(spoof_label, dtype=torch.long)
        )

def get_dataloaders(manifest_path, batch_size=16):
    """
    Creates standard dataloaders for Training, Validation, and Testing.
    """
    print(f"Initializing Multi-Task Datasets from {manifest_path.name}...")
    
    train_ds = SLPDataset(manifest_path, split="train")
    dev_ds = SLPDataset(manifest_path, split="dev")
    test_ds = SLPDataset(manifest_path, split="test")
    
    # num_workers=0 is used to avoid Windows-specific multi-processing socket errors
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, dev_loader, test_loader

# --- Verification Script ---
if __name__ == "__main__":
    csv_path = DATA_ROOT / "combined_manifest.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Please check your path.")
    else:
        train_loader, _, _ = get_dataloaders(csv_path)
        
        print("\n--- Verifying Multi-Task Batch ---")
        try:
            for waveforms, lang_labels, spoof_labels in train_loader:
                print(f"Waveform Tensor Shape: {waveforms.shape}")    # Expected: [Batch, 64000]
                print(f"Language Labels:       {lang_labels.shape}") # Expected: [Batch]
                print(f"Spoof Labels:          {spoof_labels.shape}")  # Expected: [Batch]
                
                # Check for the 4th class presence
                unique_langs = torch.unique(lang_labels).tolist()
                print(f"Unique Lang Labels in this Batch: {unique_langs}")
                
                if 3 in unique_langs:
                    print("✓ SUCCESS: Label 3 (Spam/OOD) is detected in the stream.")
                
                print("✓ SUCCESS: All tensors are LongType and ready for CrossEntropyLoss.")
                break 
        except Exception as e:
            print(f"✗ FAILED: {e}")