import os
import cv2
import glob
import yaml
import torch
import numpy as np
from facenet_pytorch import MTCNN
from tqdm import tqdm

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_and_crop(video_path, output_dir, mtcnn, frames_per_video=10, crop_size=224):
    """Extract frames from a video and crop faces."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.basename(video_path).split('.')[0]
    
    if frame_count <= 0:
        return 0
        
    # Select evenly spaced frames
    frame_indices = np.linspace(0, frame_count - 1, frames_per_video, dtype=int)
    
    os.makedirs(output_dir, exist_ok=True)
    
    saved_count = 0
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # MTCNN expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        try:
            boxes, _ = mtcnn.detect(frame_rgb)
            if boxes is not None and len(boxes) > 0:
                # Get largest face
                box = boxes[0]
                box = [int(b) for b in box]
                
                # Expand box slightly to give context
                h_img, w_img, _ = frame.shape
                x1 = max(0, box[0] - 20)
                y1 = max(0, box[1] - 20)
                x2 = min(w_img, box[2] + 20)
                y2 = min(h_img, box[3] + 20)
                
                face = frame[y1:y2, x1:x2]
                face = cv2.resize(face, (crop_size, crop_size))
                
                out_path = os.path.join(output_dir, f"{video_name}_frame_{i:03d}.jpg")
                cv2.imwrite(out_path, face)
                saved_count += 1
            else:
                # Fallback to center crop
                h, w, _ = frame.shape
                center_face = cv2.resize(frame[h//4:3*h//4, w//4:3*w//4], (crop_size, crop_size))
                out_path = os.path.join(output_dir, f"{video_name}_fallback_{i:03d}.jpg")
                cv2.imwrite(out_path, center_face)
                saved_count += 1
        except Exception as e:
            print(f"Error extracting face from {video_name} frame {i}: {e}")
            pass
            
    cap.release()
    return saved_count

def process_dataset():
    config = load_config()
    
    raw_real = config["paths"]["raw_real_dir"]
    raw_fake = config["paths"]["raw_fake_dir"]
    processed_dir = config["paths"]["processed_frames_dir"]
    
    frames_per_video = config["preprocessing"]["frames_per_video"]
    crop_size = config["preprocessing"]["face_crop_size"]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # PyTorch MPS implementation error fallback
    mtcnn = MTCNN(keep_all=False, post_process=False, device=device)
    
    for label, raw_path in [("real", raw_real), ("fake", raw_fake)]:
        print(f"Processing {label} videos from {raw_path}...")
        videos = glob.glob(os.path.join(raw_path, "*.mp4")) + glob.glob(os.path.join(raw_path, "*.avi"))
        
        out_base = os.path.join(processed_dir, label)
        
        for video_path in tqdm(videos):
            video_name = os.path.basename(video_path).split(".")[0]
            out_folder = os.path.join(out_base, video_name)
            extract_and_crop(video_path, out_folder, mtcnn, frames_per_video, crop_size)

if __name__ == "__main__":
    process_dataset()
