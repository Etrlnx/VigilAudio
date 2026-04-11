import pandas as pd
from datasets import load_dataset
from pathlib import Path
import soundfile as sf
import random

# --- Configuration ---
DATA_ROOT = Path(r"D:\College stuff\Sem6\slp\slp_data")
MASTER_CSV = DATA_ROOT / "combined_manifest.csv"
SAMPLES_PER_LANG = 500  # Updated to 500 for speed

def quick_inject_spam(csv_path):
    # 1. Load existing manifest
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    new_rows = []
    spam_configs = ["hi_in", "en_us"] 
    
    print(f"Starting Quick Injection (Limit: {SAMPLES_PER_LANG} per lang)...")

    for config in spam_configs:
        print(f"--- Fetching {config} ---")
        data_url = f"hf://datasets/google/fleurs@refs/convert/parquet/{config}/train/*.parquet"
        
        # Streaming ensures we don't load the whole 10GB+ dataset into RAM
        dataset = load_dataset("parquet", data_files=data_url, split="train", streaming=True)
        
        count = 0
        for sample in dataset:
            if count >= SAMPLES_PER_LANG:
                break
            
            # A. Path Setup
            file_name = f"{config}_spam_{count}.wav"
            relative_path = f"spam_data/{config}/{file_name}"
            absolute_path = DATA_ROOT / relative_path
            
            # B. Save Audio (Skip if already exists to save time)
            if not absolute_path.exists():
                absolute_path.parent.mkdir(parents=True, exist_ok=True)
                audio_array = sample['audio']['array']
                sr = sample['audio']['sampling_rate']
                sf.write(absolute_path, audio_array, sr)
            
            # C. Metadata
            new_rows.append({
                'path': relative_path,
                'language': 'spam_ood',
                'label': 3,
                'split': 'train', # Placeholder, shuffled below
                'source': 'fleurs_parquet',
                'spoof': 0
            })
            
            count += 1
            if count % 100 == 0:
                print(f"   Processed {count} {config} samples...")

    # 2. Final Split Logic (80/10/10)
    spam_df = pd.DataFrame(new_rows)
    indices = spam_df.index.tolist()
    random.shuffle(indices)
    
    train_end = int(0.8 * len(indices))
    dev_end = int(0.9 * len(indices))
    
    spam_df.loc[indices[:train_end], 'split'] = 'train'
    spam_df.loc[indices[train_end:dev_end], 'split'] = 'dev'
    spam_df.loc[indices[dev_end:], 'split'] = 'test'

    # 3. Append and Save
    final_df = pd.concat([df, spam_df], ignore_index=True)
    final_df.to_csv(csv_path, index=False)
    
    print(f"\n[✓] Done! Added {len(new_rows)} total spam samples.")
    print(f"[✓] Data counts: {final_df['label'].value_counts().to_dict()}")

if __name__ == "__main__":
    quick_inject_spam(MASTER_CSV)