import os
import sys
import torch
import yaml
import tempfile
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Add src and modules to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'modules')))

from model import SpatioTemporalModel
from risk_engine import RiskEngine
from image_detector import predict_image
from audio_detector import predict_audio

app = FastAPI(title="Deepfake Forensic API", version="3.0")

# --- INITIALIZATION ---
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Model
model = SpatioTemporalModel(
    sequence_length=config["model"]["sequence_length"],
    hidden_size=config["model"]["hidden_size"]
).to(device)
model_path = os.path.join(os.path.dirname(__file__), '..', config["paths"]["models_dir"], "best_model.pth")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

risk_engine = RiskEngine()

# --- MODELS ---
class ForensicResponse(BaseModel):
    prediction: str
    fake_probability: float
    threat_level: str
    risk_type: str
    recommended_action: str
    trust_score: float

@app.get("/")
def read_root():
    return {"status": "Forensic Engine Online", "version": "3.0"}

@app.post("/detect/image", response_model=ForensicResponse)
async def detect_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid image format")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        res = predict_image(tmp_path, model, device, config)
        v_prob = res["fake_probability"]
        pred = res["prediction"]
        
        risk_type = risk_engine.classify_risk(pred, file.filename)
        threat_level = risk_engine.assign_threat_level(risk_type, v_prob)
        action = risk_engine.get_recommendation(risk_type)
        
        return {
            "prediction": pred,
            "fake_probability": float(v_prob),
            "threat_level": threat_level,
            "risk_type": risk_type,
            "recommended_action": action,
            "trust_score": float((1.0 - v_prob) * 100)
        }
    finally:
        os.unlink(tmp_path)

@app.post("/detect/audio", response_model=ForensicResponse)
async def detect_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp3', '.wav')):
        raise HTTPException(status_code=400, detail="Invalid audio format")
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        res = predict_audio(tmp_path)
        v_prob = res["fake_probability"]
        pred = res["prediction"]
        
        risk_type = risk_engine.classify_risk(pred, file.filename)
        threat_level = risk_engine.assign_threat_level(risk_type, v_prob)
        action = risk_engine.get_recommendation(risk_type)
        
        return {
            "prediction": pred,
            "fake_probability": float(v_prob),
            "threat_level": threat_level,
            "risk_type": risk_type,
            "recommended_action": action,
            "trust_score": float((1.0 - v_prob) * 100)
        }
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
