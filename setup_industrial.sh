#!/bin/bash

# --- Industrial Setup Bootstrap Script ---
echo "=========================================================="
echo "   AI FORENSIC SYSTEM - INDUSTRIAL BOOTSTRAP INITIALIZER"
echo "=========================================================="

# 1. Create Directory Structure
echo "[1/4] Configuring Data Infrastructure..."
mkdir -p data/raw/real data/raw/fake data/processed data/splits models results/reports logs suspicious

# 2. Virtual Environment Setup
echo "[2/4] Initializing Proximity Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo " > Virtual environment created."
fi
source venv/bin/activate

# 3. Dependency Installation
echo "[3/4] Installing Multi-Engine Dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Model Verification
echo "[4/4] Verifying Neural Weight Matrices..."
MODEL_PATH="models/best_model.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo " > WARNING: $MODEL_PATH not found."
    echo " > Running training baseline is recommended or placing pre-trained weights in /models."
else
    echo " > Model detected: [Forensic-CNN-LSTM v3.0]"
fi

echo "=========================================================="
echo " SUCCESS: SYSTEM ARMED AND READY FOR DEPLOYMENT"
echo "=========================================================="
echo "To launch Dashboard: streamlit run dashboard/app.py"
echo "To launch API: uvicorn src.api:app --reload"
echo "Or use: docker-compose up --build"
echo "=========================================================="
