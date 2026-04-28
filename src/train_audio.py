import os
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from audio_dataset_loader import get_audio_dataloaders
from model import AudioModel

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_audio_model():
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    train_loader, test_loader = get_audio_dataloaders(config)
    if train_loader is None or len(train_loader) == 0:
        print("Empty audio dataset.")
        return
        
    model = AudioModel().to(device)
    
    all_labels = [label for _, label in train_loader.dataset]
    num_pos = sum(all_labels)
    num_neg = len(all_labels) - num_pos
    pos_weight_val = num_neg / max(1, num_pos) if num_pos > 0 else 1.0
    print(f"Audio Dataset Split: {num_neg} Authentic, {num_pos} Synthetic")
    
    criterion = nn.BCELoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    epochs = config["model"]["epochs"]
    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    
    best_loss = float('inf')
    early_stop_patience = 50
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_mfcc, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_mfcc = batch_mfcc.to(device)
            batch_labels = batch_labels.to(device).float()
            
            optimizer.zero_grad()
            probs = model(batch_mfcc)
            
            weights = torch.ones_like(batch_labels)
            weights[batch_labels == 1] = pos_weight_val
            
            loss_raw = criterion(probs, batch_labels)
            loss = (loss_raw * weights).mean()
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Audio Loss: {avg_loss:.4f}")
        
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(models_dir, "best_audio_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered")
                break
            
if __name__ == "__main__":
    train_audio_model()
