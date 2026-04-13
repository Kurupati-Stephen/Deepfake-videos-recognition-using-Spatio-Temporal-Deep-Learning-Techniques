import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import os
import yaml
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import SpatioTemporalModel

# Load config
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Load model
def load_model():
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SpatioTemporalModel(
        sequence_length=config["model"]["sequence_length"],
        hidden_size=config["model"]["hidden_size"]
    ).to(device)

    model_path = os.path.join(
        os.path.dirname(__file__), "..",
        config["paths"]["models_dir"],
        "best_model.pth"
    )

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    return model, device, config

# MAIN FUNCTION
def predict_image(image_path, model=None, device=None, config=None):
    if model is None or device is None or config is None:
        model, device, config = load_model()

    image_size = config["model"]["image_size"]
    seq_len = config["model"]["sequence_length"]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Keep original format for visual heatmap overlay
    face_img = np.array(image)
    face_img = cv2.resize(face_img, (image_size, image_size))

    img_tensor = transform(image)

    # Make copies of same image (like video frames) to satisfy spatiotemporal network
    sequence = torch.stack([img_tensor] * seq_len)
    input_tensor = sequence.unsqueeze(0).to(device)

    # Enable grad for CAM
    model.eval()
    for param in model.parameters(): 
        param.requires_grad = True
    input_tensor.requires_grad = True

    # Prediction
    v_prob, f_probs = model(input_tensor)

    score = v_prob[0]
    model.zero_grad()
    score.backward(retain_graph=True)

    gradients = model.gradients 
    activations = model.activations 

    weights = torch.mean(gradients, dim=(2, 3), keepdim=True) 
    cam = torch.sum(weights * activations, dim=1) 
    cam = F.relu(cam)
    cam = cam.detach().cpu().numpy()

    c = cam[0] # Take the first frame's cam
    c -= np.min(c)
    if np.max(c) > 0: 
        c /= np.max(c)
    cam_resized = cv2.resize(c, (image_size, image_size))

    fake_prob = v_prob.item()

    if fake_prob > 0.5:
        prediction = "Fake"
        confidence = fake_prob * 100
        status_message = "Possible manipulated image detected"
    else:
        prediction = "Real"
        confidence = (1 - fake_prob) * 100
        status_message = "Image appears authentic"

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "fake_probability": fake_prob,
        "status_message": status_message,
        "face_img": face_img,
        "cam": cam_resized
    }