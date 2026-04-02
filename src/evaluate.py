import os
import yaml
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from dataset_loader import get_dataloaders
from model import SpatioTemporalModel

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate_model():
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    _, test_loader = get_dataloaders(config)
    if test_loader is None: return
        
    model = SpatioTemporalModel(
        sequence_length=config["model"]["sequence_length"],
        hidden_size=config["model"]["hidden_size"]
    ).to(device)
    
    model_path = os.path.join(config["paths"]["models_dir"], "best_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for batch_frames, batch_labels in test_loader:
            v_probs, _ = model(batch_frames.to(device))
            preds = (v_probs > 0.5).int().cpu().numpy()
            y_pred.extend(preds)
            y_scores.extend(v_probs.cpu().numpy())
            y_true.extend(batch_labels.numpy())
            
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
    
if __name__ == "__main__":
    evaluate_model()
