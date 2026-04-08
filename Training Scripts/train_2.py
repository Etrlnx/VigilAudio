import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from dataset_2 import get_dataloaders, DATA_ROOT
from model_2 import initialize_model  # Importing your new Multi-Task Model

# --- Configuration ---
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-5
MANIFEST_PATH = DATA_ROOT / "combined_manifest.csv"
SAVE_PATH = DATA_ROOT / "best_multitask_model.pt"

# Loss Weighting for Class Imbalance (Bengali vs Assamese/Nepali/Spam)
# [Bengali, Nepali, Assamese, Spam]
LANG_WEIGHTS = torch.tensor([1.0, 7.5, 6.6, 12.0])

def train_one_epoch(model, loader, optimizer, criterion_lang, criterion_spoof, scaler, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for waveforms, lang_labels, spoof_labels in pbar:
        waveforms = waveforms.to(device)
        lang_labels = lang_labels.to(device)
        spoof_labels = spoof_labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast():
            lang_logits, spoof_logits = model(waveforms)
            
            # Calculate Dual Losses
            loss_lang = criterion_lang(lang_logits, lang_labels)
            loss_spoof = criterion_spoof(spoof_logits, spoof_labels)
            
            # Compound Loss: Forcing the model to learn both tasks simultaneously
            # You can weight these (e.g., 0.7 * loss_lang + 0.3 * loss_spoof) if one task is harder
            loss = loss_lang + loss_spoof
            
        # Backpropagation with Scaler (for RTX 5070 efficiency)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(loader)

def validate(model, loader, criterion_lang, criterion_spoof, device):
    model.eval()
    lang_correct = 0
    spoof_correct = 0
    total = 0
    
    with torch.no_grad():
        for waveforms, lang_labels, spoof_labels in loader:
            waveforms = waveforms.to(device)
            lang_labels = lang_labels.to(device)
            spoof_labels = spoof_labels.to(device)
            
            lang_logits, spoof_logits = model(waveforms)
            
            lang_preds = torch.argmax(lang_logits, dim=1)
            spoof_preds = torch.argmax(spoof_logits, dim=1)
            
            lang_correct += (lang_preds == lang_labels).sum().item()
            spoof_correct += (spoof_preds == spoof_labels).sum().item()
            total += lang_labels.size(0)
            
    return lang_correct / total, spoof_correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Multi-Task Training on {device}...")

    # 1. Initialize Model & Data
    model = initialize_model(device=device, num_langs=4)
    train_loader, dev_loader, _ = get_dataloaders(MANIFEST_PATH, batch_size=BATCH_SIZE)

    # 2. Loss Functions & Optimizer
    # We apply the weights here to handle the 21k vs 3.7k imbalance mathematically
    criterion_lang = nn.CrossEntropyLoss(weight=LANG_WEIGHTS.to(device))
    criterion_spoof = nn.CrossEntropyLoss() # Usually 50/50 split in spoof manifests
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler() # For Automatic Mixed Precision

    best_val_acc = 0.0

    # 3. Training Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_lang, criterion_spoof, scaler, device
        )
        
        lang_acc, spoof_acc = validate(model, dev_loader, criterion_lang, criterion_spoof, device)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val LID Acc: {lang_acc:.4f} | Val Spoof Acc: {spoof_acc:.4f}")

        # Save best model based on LID accuracy (or combined performance)
        if lang_acc > best_val_acc:
            best_val_acc = lang_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"[✓] Best model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()