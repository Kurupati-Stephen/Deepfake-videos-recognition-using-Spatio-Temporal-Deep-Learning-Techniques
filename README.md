# Multimodal Synthetic Media Forensics and Threat Assessment System 🛡️

A Tri-Modal (Audio, Image, Video) Digital Forensic System powering Spatio-Temporal Deep Learning with Enterprise Risk Evaluation.

## Project Overview
This project detects manipulated synthetic media (deepfakes) providing comprehensive authenticity analysis across multiple domains. It analyzes audio frequencies, visual inconsistencies in stationary images, and spatial-temporal discontinuities across video frames using a Spatio-Temporal Deep Learning Network.

## Project Objective
To build an enterprise-level, cybersecurity-oriented forensic pipeline that authenticates media inputs and brings deep transparency to the decision process via Grad-CAM tampering heatmaps and an independent audio risk engine.

## Dataset
This project uses the `DeepfakeAI_Video_Recognition` dataset sourced from the Desktop. 
> Note: If the desktop dataset folder structure lacks raw MP4 video files, a resilient preprocessing script (`dataset_inspector.py`) automatically constructs a dummy video dataset to validate the CNN+LSTM pipeline.

## Folder Structure
```
MajorProject/
├── configs/
│   └── config.yaml          # Hyperparameters and path configurations
├── data/
│   ├── raw/                 # Put your real/fake mp4 videos here
│   ├── processed/           # Extracted and face-cropped frames
│   └── splits/
├── dashboard/
│   └── app.py               # Streamlit application
├── src/
│   ├── dataset_inspector.py # Inspects dataset and generates mock data if needed
│   ├── preprocessing_pipeline.py # Extracts frames & uses MTCNN/OpenCV
│   ├── dataset_loader.py    # PyTorch data loader for spatio-temporal frames
│   ├── model.py             # CNN + LSTM Architecture
│   ├── train.py             # Model training script
│   └── evaluate.py          # Metrics evaluation (Acc, Prec, F1, AUC)
├── models/                  # Saved weights (best_model.pth)
├── results/                 # Evaluation logs and metrics
└── requirements.txt         # Dependencies
```

## Installation Steps
1. Navigate to the project directory:
   ```bash
   cd ~/Desktop/MajorProject
   ```
2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration
Edit `configs/config.yaml` to match your desktop dataset path if it changes. The variables `raw_real_dir` and `raw_fake_dir` point to where the system expects video files.

## Execution Pipeline

### 1. Data Inspection & Preprocessing
To check the dataset and begin cropping faces using MTCNN:
```bash
python src/dataset_inspector.py
python src/preprocessing_pipeline.py
```

### 2. Training
To train the CNN + LSTM model on the processed frames:
```bash
python src/train.py
```
This saves the best weights to `models/best_model.pth`.

### 3. Evaluation
To calculate accuracy, precision, recall, F1, and confusion matrix:
```bash
python src/evaluate.py
```

### 4. Interactive Dashboard (Explainable Extension)
To run the Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```
Upload a video in the web interface to view the video-level prediction and the unique **frame-by-frame explainable confidence heatmap**.

### 5. Live Virtual Broadcasting (Zoom / Google Meet)
To use the forensic system directly inside live video calls, refer to the included `Integration_Guide.md`. 
By leveraging **OBS Studio (Virtual Camera)**, you can broadcast the Streamlit Dashboard directly into Zoom, allowing participants to see real-time bounding boxes and deepfake trust scores. For live audio capture from macOS, the system integrates with **BlackHole 2ch** virtual audio driver.

## Explanation of Extension Features
The Unique Extension is an *Explainable Suspicious Frame Localization*. By analyzing the individual outputs of a frame-wise classifier combined with the LSTM sequence output, the system returns a probability score for each individual frame. The top highest-probability fake frames are highlighted to explain exactly *where and when* the manipulation occurred in the video.

## Limitations and Future Scope
- **Hardware constraints**: Heavy reliance on MTCNN face cropping and Sequence learning requires a GPU for real-time inference on 4K videos.
- **Future Enhancements**: Implement attention-based mechanisms (Vision Transformers) to provide spatial Grad-CAM heatmaps over the face itself, pinpointing precise facial artifacts.

## Recent Architectural Updates
- **Dynamic Class-Imbalance Handling**: The `train.py` pipeline now algorithmically maps `BCELoss` positive weighting to directly penalize single-class dataset bias, preventing model collapse.
- **Unclamped Pure Evaluation**: Streamlit frontend was refined to remove any artificial probability temperature scaling, presenting pure inferenced neural logits and true class probabilities mapped dynamically using isolated `model.eval()` contexts.
