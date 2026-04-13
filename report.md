# Academic Report: Real-Time Forensic Detection of Deepfake Videos

## Abstract
The rapid advancement of deep generative AI models has facilitated the creation of hyper-realistic forged media, commonly known as deepfakes. These manipulated videos pose severe threats to identity security, political stability, and misinformation spread. This project proposes a comprehensive and highly interpretable spatio-temporal deep learning approach for deepfake detection. We utilize a Convolutional Neural Network (CNN) as a robust spatial feature extractor from facial frames, fused with a Long Short-Term Memory (LSTM) network to model the temporal sequence discrepancies inherent in synthetic media. Uniquely, the system incorporates an explainable AI (XAI) extension that localizes suspicious frames, calculating frame-by-frame forgery probabilities to explain the final prediction. Experiments indicate that treating deepfake detection as a concurrent spatial and sequence-learning problem yields reliable evaluation metrics when assessed through an interactive Streamlit dashboard.

## Introduction
With the rising accessibility of manipulated media tools like FaceSwap and FaceForensics manipulation algorithms, verifying video authenticity is critical. Traditional frame-by-frame image classification models fail to perceive inter-frame jitter, blinking anomalies, or unnatural lip-syncing—temporal artifacts unique to deepfakes. By combining CNNs and LSTMs, our proposed methodology captures the hidden temporal footprint left by deepfake generative models. 

## Problem Statement
Current state-of-the-art deepfake detection systems operate as "black boxes" which output a binary REAL/FAKE classification without providing a structural justification to the user. This lack of transparency restricts adoption in critical judicial, media, and security domains where explainability is paramount.

## Objectives
1. Build a functional spatio-temporal architecture using CNN feature extractors + LSTM sequence classifiers to categorize videos.
2. Develop a preprocessing pipeline utilizing MTCNN for dynamic face localization and aligned cropping.
3. Introduce an explainable suspicious frame localization mechanism that generates a frame-wise probability index and highlights anomalous temporal segments.
4. Construct an intuitive dashboard allowing real-time video upload and forensic analysis viewing.

## Proposed Methodology & System Architecture
The application pipeline consists of four main modules:
1. **Data Pre-Processing Module**: High-speed video ingestion and frame sampling. Face detection is conducted via Multi-Task Cascaded Convolutional Networks (MTCNN), ensuring precise facial bounding boxes. 
2. **Feature Extraction Module**: Cropped images are normalized and fed into a ResNet-18 structure (CNN) stripped of its fully connected layers. This outputs a 512-dimensional dense embedding for each frame.
3. **Sequence Learning Module (LSTM)**: Sequential feature embeddings are passed to an LSTM network capable of retaining memory across the video sequence, optimizing against temporal aberrations.
4. **Risk Response & Case Management System**: A rule-based NLP Risk Engine analyzes media context to designate risk categories (e.g., Financial Fraud, Cyberbullying) and dynamically outlines Threat Levels and active Solutions. A Case Manager reliably stores all analytical payloads linking unique Case IDs internally.
5. **Explainability Dashboard Module**: The system generates a final video prediction via the hidden states of the LSTM and pushes spatial embeddings to a dense layer for frame-wise analysis. A Streamlit frontend renders the video, threat levels, trust scores, forensic analytics, and dynamic localized reports.

## Results Summary
The model was evaluated against metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Using categorical cross-entropy and tracking loss paradigms during the PyTorch training loop on pre-processed videos demonstrated that convergence was achieved. The dashboard effectively visualizes isolated artifacted frames, significantly simplifying human verification.

---

## Viva Questions and Answers

**Q1: Why did you choose a CNN + LSTM architecture over a 3D CNN (C3D)?**
*Answer:* A CNN+LSTM architecture is easier to train on standard GPUs and allows for a clearer separation of spatial artifacts (like blurring on the nose) and temporal artifacts (like jitter between frames). While 3D CNNs are powerful, they are highly parameter-dense and less interpretable on a frame-by-frame basis, making our chosen model better for generating the "explainable suspicious frames" extension.

**Q2: How does MTCNN improve your pipeline compared to classic Haar Cascades?**
*Answer:* MTCNN (Multi-Task Cascaded Convolutional Network) not only detects faces but aligns them based on 5 facial landmarks (eyes, nose, mouth corners). This alignment ensures the CNN sees consistent facial structures, greatly reducing background noise and improving deepfake classification accuracy over simple Haar Cascades.

**Q3: What do you mean by an "Explainable Extension" in your project?**
*Answer:* Instead of just saying "Fake" or "Real", the dashboard produces a frame-by-frame breakdown, pointing exactly to the timestamp/frames that caused the network to flag the video. This transparency builds trust and serves as a forensic tool rather than just a black-box classifier.

**Q4: How did you handle the situation when dataset paths are disconnected or videos are missing?**
*Answer:* The pipeline incorporates a robust `dataset_inspector.py`. If raw videos are absent, it automatically spins up a dummy dataset of synthetic sequence frames. This guarantees the PyTorch dataloaders, model graph, and Streamlit dashboard remain operative and testable during development.

**Q5: How can this system be enhanced in the future?**
*Answer:* I would introduce Vision Transformers (ViT) to replace the CNN and apply Grad-CAM (Gradient-weighted Class Activation Mapping) on the output to highlight the exact facial pixels (like the tip of the nose or eyes) that have been tampered with, rather than just identifying the frame.
