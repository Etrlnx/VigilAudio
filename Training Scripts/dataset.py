import torch
import pandas as pd
import librosa
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# --- Configuration ---
DATA_ROOT = Path(r"D:\College stuff\Sem6\slp\slp_data")

class SLPDataset(Dataset):
    def __init__(self, manifest_file, split="train", max_seconds=4, target_sr=16000):
        # Load the CSV and filter for the correct split (train/dev/test)
        df = pd.read_csv(manifest_file)
        self.df = df[df['split'] == split].reset_index(drop=True)
        
        self.max_length = max_seconds * target_sr
        self.target_sr = target_sr
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Build the absolute path to the audio file
        audio_path = str(DATA_ROOT / row['path'])
        label = row['label']
        
        try:
            # Librosa automatically handles mp3/wav, converts to mono, AND resamples to 16kHz in one step!
            # It returns a 1D numpy array.
            waveform_np, _ = librosa.load(audio_path, sr=self.target_sr, mono=True)
            
            # Convert the numpy array to a PyTorch tensor and add a channel dimension [1, length]
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
            
        except Exception as e:
            # If a file is corrupted, print the error but return a blank tensor so training continues
            print(f"[!] Librosa error loading {audio_path}: {e}")
            waveform = torch.zeros((1, self.max_length), dtype=torch.float32)
            
        # Standardize Length (Pad or Truncate)
        current_length = waveform.shape[1]
        
        if current_length > self.max_length:
            # Truncate: cut off anything past 4 seconds
            waveform = waveform[:, :self.max_length]
        elif current_length < self.max_length:
            # Pad: add zeros (silence) to the end until it reaches exactly 4 seconds
            pad_amount = self.max_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        # Remove the channel dimension so it returns a flat 1D tensor of shape [64000]
        return waveform.squeeze(0), torch.tensor(label, dtype=torch.long)

def get_dataloaders(manifest_path, batch_size=16):
    """Creates balanced dataloaders for training and testing."""
    print("Initializing Datasets...")
    train_ds = SLPDataset(manifest_path, split="train")
    dev_ds = SLPDataset(manifest_path, split="dev")
    test_ds = SLPDataset(manifest_path, split="test")
    
    # --- Handling the Class Imbalance ---
    # Calculate weights so Assamese and Nepali are picked more often
    print("Calculating class weights for balancing...")
    class_counts = train_ds.df['label'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_ds.df['label']]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    # Create DataLoaders (num_workers=0 is safer on Windows to prevent multiprocessing freezes)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, dev_loader, test_loader

# --- Quick Test ---
if __name__ == "__main__":
    csv_path = DATA_ROOT / "combined_manifest.csv"
    train_loader, dev_loader, test_loader = get_dataloaders(csv_path)
    
    print("\nExtracting a test batch...")
    for waveforms, labels in train_loader:
        print(f"Batch Waveform Shape: {waveforms.shape}")  # Should be [16, 64000]
        print(f"Batch Labels Shape: {labels.shape}")        # Should be [16]
        print(f"Label distribution (Bengali=0, Nepali=1, Assamese=2):")
        print(torch.bincount(labels, minlength=3))
        print("\nSUCCESS! The audio pipeline is fully operational.")
        break