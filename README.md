# Real-Time Forensic Detection of Deepfake Videos 🛡️

A Spatio-Temporal Deep Learning (CNN + LSTM) Deepfake Video Detection System with Explainable Suspicious Frame Localization.

## Project Overview
This project detects manipulated (Deepfake) videos by analyzing the spatial features of faces (frame-by-frame) using an advanced Convolutional Neural Network (CNN) and monitoring temporal inconsistencies across those frames using a Long Short-Term Memory (LSTM) network.

## Project Objective
To build a robust pipeline that classifies a video as REAL or FAKE, bringing transparency into the decision process through an explainable dashboard that highlights the specific suspicious frames that contributed to the final inference.

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

## Explanation of Extension Features
The Unique Extension is an *Explainable Suspicious Frame Localization*. By analyzing the individual outputs of a frame-wise classifier combined with the LSTM sequence output, the system returns a probability score for each individual frame. The top highest-probability fake frames are highlighted to explain exactly *where and when* the manipulation occurred in the video.

## Limitations and Future Scope
- **Hardware constraints**: Heavy reliance on MTCNN face cropping and Sequence learning requires a GPU for real-time inference on 4K videos.
- **Future Enhancements**: Implement attention-based mechanisms (Vision Transformers) to provide spatial Grad-CAM heatmaps over the face itself, pinpointing precise facial artifacts.
