import os
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    
    # Dynamically calculate class imbalance for weighted loss
    all_labels = [label for _, label in train_loader.dataset]
    num_pos = sum(all_labels)
    num_neg = len(all_labels) - num_pos
    pos_weight_val = num_neg / max(1, num_pos) if num_pos > 0 else 1.0
    print(f"Dataset Split: {num_neg} Authentic (0), {num_pos} Synthetic (1)")
    print(f"Setting positive class weight multiplier: {pos_weight_val:.2f}")

    # Use standard BCELoss but we will manually apply class weighting per sample
    criterion = nn.BCELoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=config["model"]["learning_rate"], weight_decay=1e-5)
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
        
        for batch_frames, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_frames = batch_frames.to(device)
            batch_labels = batch_labels.to(device).float()
            
            optimizer.zero_grad()
            v_probs, f_probs = model(batch_frames)
            
            # Create sample weight tensor
            weights = torch.ones_like(batch_labels)
            weights[batch_labels == 1] = pos_weight_val
            
            # Global loss
            loss_v_raw = criterion(v_probs, batch_labels)
            loss_v = (loss_v_raw * weights).mean()
            
            # Frame wise loss
            batch_labels_expanded = batch_labels.unsqueeze(1).expand(-1, f_probs.size(1))
            weights_expanded = weights.unsqueeze(1).expand(-1, f_probs.size(1))
            loss_f_raw = criterion(f_probs, batch_labels_expanded)
            loss_f = (loss_f_raw * weights_expanded).mean()
            
            loss = loss_v + 0.5 * loss_f
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Step the scheduler
        scheduler.step(avg_loss)
        
        # Early Stopping Logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(models_dir, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break
            
if __name__ == "__main__":
    train_model()
