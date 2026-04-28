import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from facenet_pytorch import MTCNN

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import SpatialModel

class ImageDeepfakeDetector:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.crop_size = config["model"]["image_size"]

        self.model = SpatialModel().to(self.device)
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', config["paths"]["models_dir"], "best_image_model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        mtcnn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(keep_all=False, post_process=False, device=mtcnn_device)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            return None, None, None
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        face_img = None
        
        try:
            boxes, probs = self.mtcnn.detect(frame_rgb)
            if boxes is not None and len(boxes) > 0 and probs[0] > 0.75:
                box = [int(b) for b in boxes[0]]
                x1 = max(0, box[0] - 20); y1 = max(0, box[1] - 20)
                x2 = min(w, box[2] + 20); y2 = min(h, box[3] + 20)
                face_img = frame_rgb[y1:y2, x1:x2]
        except Exception: 
            pass
            
        if face_img is None or face_img.size == 0:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_haar = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces_haar) > 0:
                (x, y, fw, fh) = faces_haar[0]
                face_img = frame_rgb[max(0, y-10):min(h, y+fh+10), max(0, x-10):min(w, x+fw+10)]
                
        if face_img is None or face_img.size == 0:
            face_img = frame_rgb[h//4:3*h//4, w//4:3*w//4]
            
        face_img = cv2.resize(face_img, (self.crop_size, self.crop_size))
        tensor_img = self.transform(face_img).unsqueeze(0).to(self.device)
        
        self.model.eval()
        for param in self.model.parameters(): param.requires_grad = True
        tensor_img.requires_grad = True
        
        prob = self.model(tensor_img)
        
        score = prob[0]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        weights = torch.mean(self.model.gradients, dim=(2, 3), keepdim=True) 
        cam = torch.sum(weights * self.model.activations, dim=1) 
        cam = torch.nn.functional.relu(cam)
        cam = cam.detach().cpu().numpy()
        
        c = cam[0]
        c -= np.min(c)
        if np.max(c) > 0: c /= np.max(c)
        c = cv2.resize(c, (self.crop_size, self.crop_size))
        
        self.model.eval()
        return prob.item(), face_img, c

def predict_image(image_path, model=None, device=None, config=None):
    # Ignoring `model` and `device` parameters from legacy SpatioTemporal pipeline
    detector = ImageDeepfakeDetector(config)
    final_prob, face_img, cam = detector.detect(image_path)
    
    if final_prob is None:
        return {
            "fake_probability": 0.0,
            "prediction": "UNKNOWN",
            "status_message": "No facial details detected in image.",
            "face_img": None,
            "cam": None
        }

    prediction = "SYNTHETIC" if final_prob > 0.5 else "AUTHENTIC"
    return {
        "fake_probability": final_prob,
        "prediction": prediction,
        "status_message": "Spatial analysis completed successfully",
        "face_img": face_img,
        "cam": cam
    }
