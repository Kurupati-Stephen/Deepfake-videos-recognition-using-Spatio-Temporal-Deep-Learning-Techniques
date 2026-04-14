# Academic Report: Multimodal Synthetic Media Forensics and Threat Assessment System

## Abstract
The rapid advancement of deep generative AI models has facilitated the creation of hyper-realistic forged media, commonly known as deepfakes. These manipulated files pose severe threats to identity security, political stability, and misinformation spread. This project proposes a comprehensive and highly interpretable **Tri-Modal Deep Learning approach** for digital forensics, evaluating Video, Image, and Audio inputs. For video and images, we utilize a Convolutional Neural Network (CNN) as a robust spatial feature extractor, fused with a Long Short-Term Memory (LSTM) network to model temporal sequence discrepancies. For audio, the system analyzes physical vocal-tract resonance parameters and frequency anomalies. Uniquely, the system incorporates an enterprise risk engine and an explainable AI (XAI) extension via Grad-CAM that visually localizes suspicious spatial artifacts, rendering a transparent threat assessment via an interactive Streamlit dashboard.

## Introduction
With the rising accessibility of manipulated media tools like FaceSwap, Voice Cloning algorithms, and Diffusion models, verifying multimedia authenticity is critical. Traditional frame-by-frame image classifiers fail to perceive inter-frame jitter or unnatural lip-syncing, while standard text-based sentiment analyzers fail to catch synthetic vocal anomalies. By combining CNNs, LSTMs, and explicit frequency domain analysis (MFCCs), our proposed methodology captures the hidden microscopic footprints left by generative multi-modal forgery models.

## Problem Statement
Current state-of-the-art deepfake detection systems operate as "black boxes" which output a binary REAL/FAKE classification on a single medium without providing a structural justification to the user. Furthermore, modern threats are multimodal (e.g., cloned audio combined with a static deepfake face). This lack of transparency and modality restriction limits adoption in critical judicial, media, and security domains where holistic explainability is paramount.

## Objectives
1. Build a functional tri-modal (Audio, Image, Video) classification architecture.
2. Develop a spatial-temporal pipeline utilizing MTCNN for dynamic face localization, feeding into a CNN+LSTM network.
3. Introduce an explicit Audio Forensics module evaluating spectral anomalies and temporal frequencies.
4. Integrate an explainable suspicious frame localization mechanism (Grad-CAM) that generates spatial heatmaps of anomalies.
5. Construct an intuitive dashboard equipped with automated Threat Assessment algorithms and incident reporting.

## Proposed Methodology & System Architecture
The application pipeline consists of five main modules:
1. **Multi-Modal Data Ingestion Module**: High-speed routing of files. Videos undergo MTCNN frame sampling; Images bypass sequence temporal limits via dynamic tensor stacking; Audio files are evaluated for acoustic characteristics.
2. **Spatial Feature Extraction Module**: Cropped images/frames are normalized and fed into a ResNet-18 structure (CNN). This outputs a dense embedding isolating unnatural pixel blending boundaries.
3. **Sequence & Audio Learning Module**: Video sequential embeddings are passed to an LSTM network optimizing against temporal aberrations. Audio files undergo distinct spectrogram/MFCC mathematical analysis.
4. **Risk Response & Case Management System**: A rule-based Risk Engine analyzes media context to designate risk categories (e.g., Financial Fraud, Cyberbullying) and dynamically formulates Threat Levels and active Solutions. A Case Manager reliably stores all analytical payloads linking unique Case IDs internally.
5. **Explainability Dashboard Module**: The Streamlit frontend renders the media, threat levels, trust scores, forensic analytics (audio waves & visual heatmaps), and downloadable localized incidence reports.

## Results Summary
The model was evaluated against metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Using categorical cross-entropy during the PyTorch training loop on pre-processed videos demonstrated high convergence. Crucially, the dashboard effectively isolates and visualizes artifacted frames or suspicious audio waveforms, significantly simplifying human verification and trust.

---

## Viva Questions and Answers

**Q1: Why did you expand the project to be "Multimodal" instead of just Video Deepfake detection like other teams?**
*Answer:* Modern threats rarely rely on just video. Audio cloning (Voice deepfakes) and high-quality image manipulation are now widely used in financial fraud and phishing. By engineering a tri-modal system, we created a comprehensive cybersecurity tool that authenticates the entire spectrum of digital media, rather than just a narrow subset of video deepfakes.

**Q2: How does the system highlight exactly where an image or video is manipulated?**
*Answer:* We implemented an "Explainable Extension" using Grad-CAM (Gradient-weighted Class Activation Mapping). Instead of a black box guessing "Fake", the dashboard traces the neural network's gradients backwards to produce a thermal heatmap pointing exactly to the specific facial pixels (e.g., a badly blended chin boundary) that triggered the alarm.

**Q3: How does MTCNN improve your visual pipeline compared to classic Haar Cascades?**
*Answer:* MTCNN (Multi-Task Cascaded Convolutional Network) not only detects faces but aligns them based on 5 facial landmarks. This alignment ensures the CNN sees consistent facial structures, greatly reducing background noise and improving spatial deepfake classification accuracy over simple Haar Cascades.

**Q4: How did you handle Audio detection?**
*Answer:* We implemented a discrete audio pipeline that uses the `librosa` library to extract physical characteristics like Mel-Frequency Cepstral Coefficients (MFCCs), spectral centroids, and zero-crossing rates. These frequency-domain markers easily expose the unnatural biological tracts present in cloned, synthetic voice generation.

**Q5: What is the purpose of the Threat Assessment and Case Management Engine?**
*Answer:* We wanted to elevate this from a basic machine learning script to an Enterprise-grade security product. The engine assigns a specific Case ID and calculates a "Trust Score" and "Threat Level" (Low/Medium/Critical) based on confidence logs and user context, creating actionable reports rather than just raw probabilities.
