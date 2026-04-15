import torch
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from transformers import Wav2Vec2ForSequenceClassification
from dataset import get_dataloaders, DATA_ROOT

warnings.filterwarnings("ignore")

# --- Configuration ---
BATCH_SIZE = 8
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
MODEL_WEIGHTS = DATA_ROOT / "best_lid_model.pt"

# The two parallel datasets
PRISTINE_MANIFEST = DATA_ROOT / "combined_manifest.csv"
SPOOFED_MANIFEST = DATA_ROOT / "spoofed_manifest.csv"

# Ensure this matches your dataset label mapping exactly!
LANG_NAMES = ["Bengali", "Nepali", "Assamese"] 

def evaluate_and_plot(model, device, manifest_path, report_title, cmap_color, output_filename):
    """Runs inference, prints an F1 report, and saves a confusion matrix heatmap."""
    print(f"\nLoading test data from: {manifest_path.name}...")
    
    # We only need the test_loader (the 3rd item returned by your dataset pipeline)
    _, _, test_loader = get_dataloaders(manifest_path, batch_size=BATCH_SIZE)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Evaluating {report_title}")
        for waveforms, labels in pbar:
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(waveforms)
                logits = outputs.logits
                
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # --- 1. Terminal Output: Classification Report ---
    print("\n" + "="*50)
    print(f"{report_title.upper()} CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(all_labels, all_preds, target_names=LANG_NAMES, digits=4)
    print(report)
    
    # --- 2. File Output: Confusion Matrix Image ---
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_color, 
                xticklabels=LANG_NAMES, yticklabels=LANG_NAMES)
    plt.title(f'{report_title} Confusion Matrix')
    plt.ylabel('Actual Language')
    plt.xlabel('Predicted Language')
    
    plot_path = DATA_ROOT / output_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close() # Critical: close the plot so the next one doesn't draw on top of it!
    
    print(f"[+] Matrix successfully saved to: {plot_path}")

def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Deploying Baseline Model on {device}...")

    # Load Model Architecture and Custom Weights
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model.to(device)
    model.eval() # Lock dropout and batchnorm layers for inference

    # ==========================================
    # PHASE 1: PRISTINE BASELINE EVALUATION
    # ==========================================
    print("\n" + "#"*50)
    print("PHASE 1: BASELINE EVALUATION (PRISTINE AUDIO)")
    print("#"*50)
    evaluate_and_plot(
        model=model, 
        device=device, 
        manifest_path=PRISTINE_MANIFEST, 
        report_title="Pristine Baseline", 
        cmap_color="Blues", 
        output_filename="pristine_confusion_matrix.png"
    )

    # ==========================================
    # PHASE 2: SPOOFING ATTACK EVALUATION
    # ==========================================
    print("\n" + "#"*50)
    print("PHASE 2: VULNERABILITY EVALUATION (SPOOFED AUDIO)")
    print("#"*50)
    evaluate_and_plot(
        model=model, 
        device=device, 
        manifest_path=SPOOFED_MANIFEST, 
        report_title="Spoofing Attack", 
        cmap_color="Reds", # Using red to visually indicate the attack state
        output_filename="spoofed_confusion_matrix.png"
    )
    
    print("\n" + "="*50)
    print("[✓] Dual evaluation complete. Check your slp_data folder for the two matrix plots!")
    print("="*50)

if __name__ == "__main__":
    main()