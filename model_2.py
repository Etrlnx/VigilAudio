import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config

class MultiTaskWav2Vec2(nn.Module):
    def __init__(self, model_name_or_path, num_langs=4, num_spoof=2, dropout=0.1):
        super(MultiTaskWav2Vec2, self).__init__()
        
        # 1. THE CORE: XLSR-53 Backbone
        # This extracts the high-level acoustic features (embeddings)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name_or_path)
        
        # 2. FEATURE EXTRACTOR FREEZING
        # The first few layers of Wav2Vec2 process raw waveforms into spectrograms.
        # These are usually stable, so we freeze them to save VRAM and prevent 'forgetting'.
        self.wav2vec2.feature_extractor._freeze_parameters()
        
        config = self.wav2vec2.config
        hidden_size = config.hidden_size # 1024 for Large XLS-R
        
        # 3. TASK-SPECIFIC LAYERS (The 'Heads')
        # We add an intermediate layer (512) to help the model learn 
        # complex relations like 'Tone' and 'Intent'.
        
        # Head A: Language ID + Spam/OOD
        self.language_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512), # Added for training stability
            nn.GELU(),         # Advanced activation used in Transformers
            nn.Dropout(dropout),
            nn.Linear(512, num_langs)
        )
        
        # Head B: Security/Spoof Detection
        self.spoof_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_spoof)
        )

    def forward(self, input_values, attention_mask=None):
        """
        Args:
            input_values: [batch, 64000] (4 seconds of audio)
            attention_mask: Optional mask for variable length audio
        """
        # Pass through Wav2Vec2
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        
        # Get the sequence of hidden states: [batch, seq_len, 1024]
        hidden_states = outputs.last_hidden_state
        
        # POOLING STRATEGY
        # Instead of just taking the first token, we mean-pool across the entire
        # 4 seconds. This captures the 'background noise' (Spam) and 
        # 'transducer artifacts' (Spoof) much better.
        if attention_mask is not None:
            # Masked mean pooling to ignore padding
            active_elements = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states.device).float()
            sum_embeddings = torch.sum(hidden_states * active_elements, dim=1)
            count_elements = torch.clamp(active_elements.sum(1), min=1e-9)
            pooled_output = sum_embeddings / count_elements
        else:
            pooled_output = torch.mean(hidden_states, dim=1)
        
        # Final Classifications
        lang_logits = self.language_head(pooled_output)
        spoof_logits = self.spoof_head(pooled_output)
        
        return lang_logits, spoof_logits

# --- Utility Functions for train.py ---

def initialize_model(model_path="facebook/wav2vec2-large-xlsr-53", device="cuda", num_langs=4):
    model = MultiTaskWav2Vec2(model_path, num_langs=num_langs) # Pass 4 here
    model.to(device)
    
    # Calculate parameter count (to show complexity)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Initialized. Total Params: {total_params:,} | Trainable: {trainable_params:,}")
    
    return model

if __name__ == "__main__":
    # Sanity Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = initialize_model(device=device)
    
    # Simulate a batch: 2 audio clips, 4 seconds each
    dummy_audio = torch.randn(2, 64000).to(device)
    l_out, s_out = model(dummy_audio)
    
    print(f"Output Check:")
    print(f"-> Language logits: {l_out.shape} (Expected [2, 4])")
    print(f"-> Spoof logits:    {s_out.shape} (Expected [2, 2])")