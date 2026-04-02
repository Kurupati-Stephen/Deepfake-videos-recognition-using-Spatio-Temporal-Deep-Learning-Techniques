import os
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from dataset_loader import get_dataloaders
from model import SpatioTemporalModel

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_model():
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    train_loader, test_loader = get_dataloaders(config)
    if train_loader is None:
        print("Empty dataset.")
        return
        
    model = SpatioTemporalModel(
        sequence_length=config["model"]["sequence_length"],
        hidden_size=config["model"]["hidden_size"]
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=config["model"]["learning_rate"])
    
    epochs = config["model"]["epochs"]
    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_frames, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_frames = batch_frames.to(device)
            batch_labels = batch_labels.to(device).float()
            
            optimizer.zero_grad()
            v_probs, f_probs = model(batch_frames)
            
            # Global loss
            loss_v = criterion(v_probs, batch_labels)
            
            # Frame wise loss
            batch_labels_expanded = batch_labels.unsqueeze(1).expand(-1, f_probs.size(1))
            loss_f = criterion(f_probs, batch_labels_expanded)
            
            loss = loss_v + 0.5 * loss_f
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(models_dir, "best_model.pth"))
            
if __name__ == "__main__":
    train_model()
