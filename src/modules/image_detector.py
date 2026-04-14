import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from facenet_pytorch import MTCNN

# We reuse the SpatioTemporalModel to evaluate spatial inconsistencies on the single frame.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import SpatioTemporalModel

class ImageDeepfakeDetector:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        self.seq_len = config["model"]["sequence_length"]
        self.crop_size = config["model"]["image_size"]

        # Load Existing Spatial-Temporal Model
        self.model = SpatioTemporalModel(
            sequence_length=self.seq_len,
            hidden_size=config["model"]["hidden_size"]
        ).to(self.device)
        
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', config["paths"]["models_dir"], "best_model.pth")
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
        """ Processes single image and returns fake probability, face image, and gradcam. """
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
            # Fallback center crop
            face_img = frame_rgb[h//4:3*h//4, w//4:3*w//4]
            
        face_img = cv2.resize(face_img, (self.crop_size, self.crop_size))
        tensor_img = self.transform(face_img)
        
        # Duplicate the image sequence_length times to appease the LSTM
        tensor_seq = torch.stack([tensor_img] * self.seq_len).unsqueeze(0).to(self.device)
        
        # Run GradCAM specifically on the single spatial frame
        self.model.eval()
        for param in self.model.parameters(): param.requires_grad = True
        tensor_seq.requires_grad = True
        
        v_prob, f_probs = self.model(tensor_seq)
        
        # We rely heavily on the spatial frame confidence
        avg_f_prob = torch.mean(f_probs).item()
        
        # Extract GradCAM using 1st frame
        score = v_prob[0]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        weights = torch.mean(self.model.gradients, dim=(2, 3), keepdim=True) 
        cam = torch.sum(weights * self.model.activations, dim=1) 
        cam = torch.nn.functional.relu(cam)
        cam = cam.detach().cpu().numpy()
        
        # Use first frame CAM
        c = cam[0]
        c -= np.min(c)
        if np.max(c) > 0: c /= np.max(c)
        c = cv2.resize(c, (self.crop_size, self.crop_size))
        
        self.model.eval()
        
        # In a purely spatial test, we mix LSTM overall probability with the spatial frame probability
        final_prob = (v_prob.item() + avg_f_prob) / 2.0
        
        return final_prob, face_img, c
