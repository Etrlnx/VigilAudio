import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from dataset_2 import get_dataloaders, DATA_ROOT
from model_2 import initialize_model

# --- Configuration ---
BATCH_SIZE = 8
MODEL_WEIGHTS = DATA_ROOT / "best_multitask_model.pt"
MANIFEST = DATA_ROOT / "combined_manifest.csv"

# Updated to include the Spam/OOD class
LANG_NAMES = ["Bengali", "Nepali", "Assamese", "Spam/OOD"]
SPOOF_NAMES = ["Bona Fide", "Spoofed"]

def calculate_eer(y_true, y_score):
    """Calculates EER with safety checks for NaN or constant values."""
    # Convert to numpy arrays and filter out any NaNs
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    mask = ~np.isnan(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(np.unique(y_true)) < 2:
        print("⚠️ Warning: Only one class present in spoof labels. EER cannot be calculated.")
        return 0.0

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    
    # Safety check for empty or invalid ROC curve
    if len(fpr) < 2:
        return 0.0

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


def run_evaluation(model, device, loader):
    model.eval()
    
    results = {
        "lang_preds": [], "lang_labels": [],
        "spoof_preds": [], "spoof_labels": [],
        "spoof_scores": [] # Probabilities for EER calculation
    }
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Running Inference")
        for waveforms, lang_labels, spoof_labels in pbar:
            waveforms = waveforms.to(device)
            
            with torch.cuda.amp.autocast():
                lang_logits, spoof_logits = model(waveforms)
                
                # Get probabilities for the 'Spoofed' class (index 1)
                spoof_probs = torch.softmax(spoof_logits, dim=1)[:, 1]
            
            results["lang_preds"].extend(torch.argmax(lang_logits, dim=1).cpu().numpy())
            results["lang_labels"].extend(lang_labels.numpy())
            results["spoof_preds"].extend(torch.argmax(spoof_logits, dim=1).cpu().numpy())
            results["spoof_labels"].extend(spoof_labels.numpy())
            results["spoof_scores"].extend(spoof_probs.cpu().numpy())
            
    return results

def plot_cm(y_true, y_pred, labels, title, filename, color):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=color, 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    path = DATA_ROOT / filename
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] {title} Matrix saved to: {path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Custom MTL Model
    model = initialize_model(device=device, num_langs=4)
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    
    # 2. Get Test Data
    _, _, test_loader = get_dataloaders(MANIFEST, batch_size=BATCH_SIZE)
    
    # 3. Execute Inference
    res = run_evaluation(model, device, test_loader)
    
    # --- TASK 1: LANGUAGE IDENTIFICATION & SPAM REPORT ---
    print("\n" + "="*60)
    print("TASK 1: LANGUAGE ID & OOD REJECTION (MACRO F1)")
    print("="*60)
    # Using Macro average to treat Assamese and Bengali with equal weight
    print(classification_report(res["lang_labels"], res["lang_preds"], 
                                target_names=LANG_NAMES, digits=4))
    
    plot_cm(res["lang_labels"], res["lang_preds"], LANG_NAMES, 
            'LID & Spam Confusion Matrix', 'lang_confusion_matrix.png', 'Blues')

    # --- TASK 2: SPOOF DETECTION & SECURITY REPORT ---
    print("\n" + "="*60)
    print("TASK 2: SPOOF DETECTION (SECURITY METRICS)")
    print("="*60)
    eer = calculate_eer(res["spoof_labels"], res["spoof_scores"])
    print(f"Equal Error Rate (EER): {eer:.4%}")
    print(f"(Note: Lower EER means better liveness detection)")
    
    print("\n" + classification_report(res["spoof_labels"], res["spoof_preds"], 
                                    target_names=SPOOF_NAMES, digits=4))
    
    plot_cm(res["spoof_labels"], res["spoof_preds"], SPOOF_NAMES, 
            'Spoof Detection Confusion Matrix', 'spoof_confusion_matrix.png', 'Reds')

if __name__ == "__main__":
    main()