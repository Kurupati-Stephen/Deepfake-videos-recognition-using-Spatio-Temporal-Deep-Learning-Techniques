import os
import yaml
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from dataset_loader import get_dataloaders
from model import SpatioTemporalModel

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

from collections import Counter
import numpy as np

def evaluate_model():
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    train_loader, test_loader = get_dataloaders(config)
    if test_loader is None or len(test_loader) == 0: 
        print("Test loader is empty or None!")
        return
        
    model = SpatioTemporalModel(
        sequence_length=config["model"]["sequence_length"],
        hidden_size=config["model"]["hidden_size"]
    ).to(device)
    
    model_path = os.path.join(config["paths"]["models_dir"], "best_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print(f"Warning: Model not found at {model_path}. Using untrained random weights.")
    model.eval()
    
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for batch_frames, batch_labels in test_loader:
            v_probs, _ = model(batch_frames.to(device))
            # v_probs is from sigmoid, range [0, 1]
            preds = (v_probs > 0.5).int().cpu().flatten().numpy()
            v_probs_flat = v_probs.cpu().flatten().numpy()
            
            y_pred.extend(preds.tolist())
            y_scores.extend(v_probs_flat.tolist())
            y_true.extend(batch_labels.cpu().flatten().numpy().tolist())
            
    if len(y_true) == 0:
        print("Cannot evaluate. No valid evaluation samples loaded.")
        return
        
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    print(f"DEBUG | y_true distribution: {dict(Counter(y_true))}")
    print(f"DEBUG | y_pred distribution: {dict(Counter(y_pred))}")
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"DEBUG | Confusion Matrix:\n{cm}")
        
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    out_str = f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\n"
    out_str += f"\n--- DEBUG INFO ---\n"
    out_str += f"y_true distrib: {dict(Counter(y_true))}\n"
    out_str += f"y_pred distrib: {dict(Counter(y_pred))}\n"
    out_str += f"Total Samples: {len(y_true)}\n"
    
    print(out_str)
    
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)
    with open(os.path.join(config["paths"]["results_dir"], "metrics.txt"), "w") as f:
        f.write(out_str)

if __name__ == "__main__":
    evaluate_model()
