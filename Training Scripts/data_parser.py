import os
import tarfile
import shutil
import pandas as pd
from pathlib import Path

# --- Configuration ---
DATA_ROOT = Path(r"./")
PROCESSED_DIR = DATA_ROOT / "processed_data"
OUTPUT_CSV = DATA_ROOT / "combined_manifest.csv"

LANG_MAP = {"bengali": 0, "nepali": 1, "assamese": 2}

def setup_folders():
    """Creates the proper train, dev, and test folders."""
    print("[+] Setting up split folders...")
    for split in ["train", "dev", "test"]:
        (PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)

def process_mcv():
    """Extracts and routes MCV audio based on train/dev/test TSVs."""
    records = []
    mcv_codes = {"bn": "bengali", "ne": "nepali", "as": "assamese"}
    
    for code, full_name in mcv_codes.items():
        tar_path = DATA_ROOT / "mcv_raw" / f"{code}.tar.gz"
        if not tar_path.exists():
            print(f"[-] Missing MCV file: {tar_path}")
            continue
            
        print(f"[+] Parsing MCV {full_name} TSVs...")
        split_map = {} # Maps filename to its proper split folder
        
        try:
            # First Pass: Read TSVs to map filenames to splits
            with tarfile.open(tar_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith((".tsv")):
                        # Determine if this TSV is for train, dev, or test
                        split = None
                        if "train.tsv" in member.name: split = "train"
                        elif "dev.tsv" in member.name: split = "dev"
                        elif "test.tsv" in member.name: split = "test"
                        
                        if split:
                            f = tar.extractfile(member)
                            df = pd.read_csv(f, sep='\t', low_memory=False)
                            path_col = next((c for c in df.columns if c.lower() in ['path', 'filename']), None)
                            if path_col:
                                for p in df[path_col]:
                                    split_map[Path(p).name] = split

            # Second Pass: Extract only the needed audio files to proper folders
            print(f"[+] Extracting mapped audio for MCV {full_name} (This may take a moment)...")
            extracted_count = 0
            with tarfile.open(tar_path, "r:gz") as tar:
                for member in tar.getmembers():
                    basename = Path(member.name).name
                    if basename in split_map:
                        split = split_map[basename]
                        
                        # Prefix filename to prevent overwriting
                        out_name = f"{code}_mcv_{basename}"
                        out_path = PROCESSED_DIR / split / out_name
                        
                        # Extract the file
                        source = tar.extractfile(member)
                        if source:
                            with open(out_path, "wb") as target:
                                shutil.copyfileobj(source, target)
                                
                            records.append({
                                "path": f"processed_data/{split}/{out_name}",
                                "language": full_name,
                                "label": LANG_MAP[full_name],
                                "split": split,
                                "source": "mcv"
                            })
                            extracted_count += 1
                            if extracted_count % 1000 == 0:
                                print(f"    ...extracted {extracted_count} files")
                                
        except Exception as e:
            print(f"[!] Error processing MCV {full_name}: {e}")
            
    return records

def process_fleurs():
    """Extracts FLEURS audio directly from the pre-split tar.gz files."""
    records = []
    fleurs_dirs = {"fluer_as": "assamese", "fluer_ne": "nepali"}
    
    for folder, full_name in fleurs_dirs.items():
        dir_path = DATA_ROOT / folder
        if not dir_path.exists():
            print(f"[-] Missing FLEURS folder: {dir_path}")
            continue
            
        print(f"[+] Extracting FLEURS {full_name}...")
        for split in ["train", "dev", "test"]:
            tar_path = dir_path / f"{split}.tar.gz"
            if not tar_path.exists(): continue
            
            try:
                extracted_count = 0
                with tarfile.open(tar_path, "r:gz") as tar:
                    for member in tar.getmembers():
                        # Look for actual audio files inside the tar
                        if member.isreg() and member.name.endswith(('.wav', '.mp3')):
                            basename = Path(member.name).name
                            
                            # Prefix filename
                            out_name = f"{full_name[:2]}_fluer_{basename}"
                            out_path = PROCESSED_DIR / split / out_name
                            
                            source = tar.extractfile(member)
                            if source:
                                with open(out_path, "wb") as target:
                                    shutil.copyfileobj(source, target)
                                
                                records.append({
                                    "path": f"processed_data/{split}/{out_name}",
                                    "language": full_name,
                                    "label": LANG_MAP[full_name],
                                    "split": split,
                                    "source": "fleurs"
                                })
                                extracted_count += 1
                                if extracted_count % 500 == 0:
                                    print(f"    ...extracted {extracted_count} {split} files")
            except Exception as e:
                print(f"[!] Error processing FLEURS {full_name} {split}: {e}")
                
    return records

def main():
    print("="*40)
    print("Starting Data Parser & Extractor")
    print("="*40)
    
    setup_folders()
    
    all_data = []
    all_data.extend(process_mcv())
    all_data.extend(process_fleurs())
    
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(OUTPUT_CSV, index=False)
        print("\n" + "="*40)
        print(f"SUCCESS! Extracted and mapped {len(df)} total audio files.")
        print(f"Data saved to: {PROCESSED_DIR}")
        print("\nDataset Breakdown:")
        print(pd.crosstab(df['language'], df['split']))
        print("="*40)
    else:
        print("\n[!] No data was extracted. Please check your folder structure.")

if __name__ == "__main__":
    main()