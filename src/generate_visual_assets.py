import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dataset_loader import get_dataloaders
from model import SpatioTemporalModel
from tqdm import tqdm

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_evaluation():
    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # We want more than just 2 samples if possible, so let's load the entire dataset for plotting
    from dataset_loader import DeepfakeDataset
    from torchvision import transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config["model"]["image_size"], config["model"]["image_size"])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = DeepfakeDataset(config["paths"]["processed_frames_dir"], 
                                  sequence_length=config["model"]["sequence_length"], 
                                  transform=transform)
    
    if len(full_dataset) == 0:
        print("Dataset is empty. Cannot generate visuals.")
        return None, None, None

    loader = DataLoader(full_dataset, batch_size=config["model"]["batch_size"], shuffle=False)
    
    model = SpatioTemporalModel(
        sequence_length=config["model"]["sequence_length"],
        hidden_size=config["model"]["hidden_size"]
    ).to(device)
    
    model_path = os.path.join(config["paths"]["models_dir"], "best_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    y_true, y_pred, y_probs = [], [], []
    
    print(f"Evaluating {len(full_dataset)} samples...")
    with torch.no_grad():
        for batch_frames, batch_labels in tqdm(loader):
            v_probs, _ = model(batch_frames.to(device))
            preds = (v_probs > 0.5).int().cpu().flatten().numpy()
            probs = v_probs.cpu().flatten().numpy()
            
            y_pred.extend(preds.tolist())
            y_probs.extend(probs.tolist())
            y_true.extend(batch_labels.cpu().flatten().numpy().tolist())
            
    return np.array(y_true), np.array(y_pred), np.array(y_probs)

def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 6), facecolor='#0b1121')
    ax = plt.gca()
    ax.set_facecolor('#0b1121')
    
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', color='white', fontsize=16, pad=20)
    plt.colorbar(im)
    
    classes = ['Authentic', 'Synthetic']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, color='white', rotation=45)
    plt.yticks(tick_marks, classes, color='white')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Class', color='white')
    plt.xlabel('Predicted Class', color='white')
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#0b1121')
    plt.close()

def plot_metrics_bar(metrics, save_path):
    names = list(metrics.keys())
    values = [metrics[k] * 100 for k in names]
    
    plt.figure(figsize=(10, 6), facecolor='#0b1121')
    ax = plt.gca()
    ax.set_facecolor('#0b1121')
    
    colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#14b8a6']
    bars = plt.bar(names, values, color=colors)
    
    plt.title('Global Performance Metrics (%)', color='white', fontsize=16, pad=20)
    plt.ylim(0, 110)
    plt.yticks(color='white')
    plt.xticks(color='white')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', color='white', fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.3, color='#475569')
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#0b1121')
    plt.close()

def plot_score_distribution(y_true, y_probs, save_path):
    plt.figure(figsize=(10, 6), facecolor='#0b1121')
    ax = plt.gca()
    ax.set_facecolor('#0b1121')
    
    real_scores = y_probs[y_true == 0]
    fake_scores = y_probs[y_true == 1]
    
    plt.hist(real_scores, bins=20, alpha=0.6, label='Authentic', color='#22c55e', density=True)
    plt.hist(fake_scores, bins=20, alpha=0.6, label='Synthetic', color='#ef4444', density=True)
    
    plt.title('Prediction Score Distribution', color='white', fontsize=16, pad=20)
    plt.xlabel('Probability of being Synthetic', color='white')
    plt.ylabel('Density', color='white')
    plt.legend()
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.grid(alpha=0.2, color='#475569')
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#0b1121')
    plt.close()

def plot_class_balance(y_true, save_path):
    plt.figure(figsize=(8, 6), facecolor='#0b1121')
    ax = plt.gca()
    ax.set_facecolor('#0b1121')
    
    counts = np.bincount(y_true)
    labels = ['Authentic', 'Synthetic']
    
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#22c55e', '#ef4444'], 
            textprops={'color':"white", 'fontsize': 12}, startangle=140)
    plt.title('Dataset Class Distribution', color='white', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#0b1121')
    plt.close()

def main():
    assets_dir = "assets/visuals"
    os.makedirs(assets_dir, exist_ok=True)
    
    y_true, y_pred, y_probs = run_evaluation()
    
    if y_true is None:
        return
        
    # Metrics
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    print("\n--- PERFORMANCE SUMMARY ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    cm = confusion_matrix(y_true, y_pred)
    
    # Generate Visuals
    print("\nGenerating visual assets...")
    plot_confusion_matrix(cm, os.path.join(assets_dir, "confusion_matrix.png"))
    plot_metrics_bar(metrics, os.path.join(assets_dir, "metrics_bar.png"))
    plot_score_distribution(y_true, y_probs, os.path.join(assets_dir, "score_distribution.png"))
    plot_class_balance(y_true, os.path.join(assets_dir, "class_balance.png"))
    
    print(f"All assets saved to {assets_dir}/")

if __name__ == "__main__":
    main()
