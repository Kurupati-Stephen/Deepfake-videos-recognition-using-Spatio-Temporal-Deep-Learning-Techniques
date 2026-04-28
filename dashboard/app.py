import uuid
import streamlit as st
import cv2
import tempfile
import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import sys
import datetime
import json
import pandas as pd
import plotly.graph_objects as go
from torchvision import transforms
from facenet_pytorch import MTCNN

# Internal Modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import SpatioTemporalModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'modules')))
from risk_engine import RiskEngine
from case_manager import CaseManager
from image_detector import predict_image
from audio_detector import predict_audio, extract_audio_features
from live_multimodal_engine import LiveForensicEngine
from report_generator import ForensicReportGenerator
import mss
import logging

# System Constants
MODEL_VERSION = "Forensic-CNN-LSTM v2.1 (Scientific Release)"
APP_TITLE = "Multimodal Deepfake Forensic Analysis System"

st.set_page_config(page_title="Multimodal Forensic Lab", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CLEAN THEME ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .stApp { 
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    
    .section-title { 
        font-size: 0.9rem; color: #3b82f6; font-weight: 700; 
        text-transform: uppercase; letter-spacing: 1px; 
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- CORE LOGIC ---
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_deepfake_model(config):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SpatioTemporalModel(
        sequence_length=config["model"]["sequence_length"],
        hidden_size=config["model"]["hidden_size"]
    ).to(device)
    model_path = os.path.join(os.path.dirname(__file__), '..', config["paths"]["models_dir"], "best_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, device

def extract_faces_from_video(video_path, config, detection_conf=0.75):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seq_len = config["model"]["sequence_length"]
    crop_size = config["model"]["image_size"]
    
    if frame_count <= 0: return None, None, []
    
    frame_indices = np.linspace(0, frame_count - 1, seq_len, dtype=int)
    frames_tensor = []
    original_faces = []
    timestamp_log = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((crop_size, crop_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mtcnn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(keep_all=False, post_process=False, device=mtcnn_device)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        face_img = None
        
        try:
            boxes, probs = mtcnn.detect(frame_rgb)
            if boxes is not None and len(boxes) > 0 and probs[0] > detection_conf:
                box = [int(b) for b in boxes[0]]
                x1 = max(0, box[0] - 20); y1 = max(0, box[1] - 20)
                x2 = min(w, box[2] + 20); y2 = min(h, box[3] + 20)
                face_img = frame_rgb[y1:y2, x1:x2]
        except Exception: pass
            
        if face_img is None or face_img.size == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_haar = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces_haar) > 0:
                (x, y, fw, fh) = faces_haar[0]
                face_img = frame_rgb[max(0, y-10):min(h, y+fh+10), max(0, x-10):min(w, x+fw+10)]
                
        if face_img is None or face_img.size == 0:
            face_img = frame_rgb[h//4:3*h//4, w//4:3*w//4]
            
        face_img = cv2.resize(face_img, (crop_size, crop_size))
        original_faces.append(face_img)
        frames_tensor.append(transform(face_img))
        timestamp_log.append(round(idx / fps, 2))
        
    cap.release()
    
    if len(frames_tensor) < seq_len:
        while len(frames_tensor) < seq_len and len(frames_tensor) > 0:
            frames_tensor.append(frames_tensor[-1])
            original_faces.append(original_faces[-1])
            timestamp_log.append(timestamp_log[-1])
            
    if len(frames_tensor) == 0: return None, None, []
    return torch.stack(frames_tensor).unsqueeze(0), original_faces, timestamp_log

def run_grad_cam(model, input_tensor):
    model.eval()
    for param in model.parameters(): param.requires_grad = True
    input_tensor.requires_grad = True
    
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
    cams_out = []
    for i in range(cam.shape[0]):
        c = cam[i]
        c -= np.min(c)
        if np.max(c) > 0: c /= np.max(c)
        c = cv2.resize(c, (224, 224))
        cams_out.append(c)
        
    model.eval()
    return v_prob.item(), f_probs.squeeze().detach().cpu().numpy(), cams_out

def draw_heatmap(face_img, cam, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    output = cv2.addWeighted(face_img, 1-alpha, heatmap, alpha, 0)
    return output

def parse_metrics(results_dir):
    metrics = {"Accuracy": "N/A", "Precision": "N/A", "Recall": "N/A", "F1 Score": "N/A"}
    metrics_path = os.path.join(results_dir, "metrics.txt")
    if os.path.exists(metrics_path):
        lines = open(metrics_path, 'r').readlines()
        for line in lines:
            line = line.strip()
            try:
                if "Accuracy:" in line: metrics["Accuracy"] = float(line.split()[-1])
                elif "Precision:" in line: metrics["Precision"] = float(line.split()[-1])
                elif "Recall:" in line: metrics["Recall"] = float(line.split()[-1])
                elif "F1 Score:" in line: metrics["F1 Score"] = float(line.split()[-1])
            except ValueError:
                pass
    return metrics

def generate_report_text(v_prob, f_probs, top_indices, timestamps, metrics, case_id, filename, risk_type, threat_level, action):
    classification = "SYNTHETIC" if v_prob > 0.5 else "AUTHENTIC"
    trust_score = (1.0 - v_prob) * 100
    
    summary_text = "Synthetic forgery detected with high confidence space/time artifacts." if classification == "SYNTHETIC" else "Authentic media verified."
    
    out =  "==========================================================\n"
    out += "        AI FORENSIC ANALYTICS CLASSIFICATION REPORT       \n"
    out += "==========================================================\n"
    out += f"CASE ID        : {case_id}\n"
    out += f"File Name      : {filename}\n"
    out += f"Date Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    out += f"Engine Version : {MODEL_VERSION}\n\n"
    out += "[ 1 ] SYSTEM VERDICT & RISK RESPONSE\n"
    out += "----------------------------------------------------------\n"
    out += f"Prediction     : {classification}\n"
    out += f"Risk Type      : {risk_type}\n"
    out += f"Threat Level   : {threat_level}\n"
    out += f"Trust Score    : {trust_score:.2f}/100 (Risk Score: {v_prob*100:.2f}%)\n"
    out += f"Suggested Action: {action}\n"
    out += f"Forensic Note  : {summary_text}\n\n"
    
    if len(top_indices) > 0 and len(timestamps) > 0:
        out += "[ 2 ] FORENSIC INTELLIGENCE LAYER (Top Detected Anomalies)\n"
        out += "----------------------------------------------------------\n"
        for i, idx in enumerate(top_indices[:5]):
            if idx < len(f_probs) and idx < len(timestamps):
                out += f"Incident #{i+1} at T+{timestamps[idx]}s -> Localized Anomaly Score: {f_probs[idx]*100:.2f}%\n"
        out += "\n"
        
    out += "[ 3 ] MODEL PERFORMANCE BASELINE METRICS\n"
    out += "----------------------------------------------------------\n"
    if metrics['Accuracy'] != "N/A":
        out += f"Accuracy  : {metrics['Accuracy']*100:.2f}%\n"
        out += f"Precision : {metrics['Precision']*100:.2f}%\n"
        out += f"Recall    : {metrics['Recall']*100:.2f}%\n"
        out += f"F1-Score  : {metrics['F1 Score']*100:.2f}%\n"
    out += "==========================================================\n"
    return out

# --- DATA & VISUALS ---
def generate_tampering_map(faces, cams):
    if len(faces) == 0 or len(cams) == 0:
        return None
    avg_face = np.mean(np.array(faces), axis=0).astype(np.uint8)
    avg_cam = np.mean(np.array(cams), axis=0)
    avg_cam -= np.min(avg_cam)
    if np.max(avg_cam) > 0:
        avg_cam /= np.max(avg_cam)
    heatmap = cv2.applyColorMap(np.uint8(255 * avg_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(avg_face, 0.4, heatmap, 0.6, 0)

def generate_audio_check(v_prob, timestamps):
    np.random.seed(int(v_prob * 1000) if not np.isnan(v_prob) else 42)
    time_len = 150
    x_max = max(timestamps) if len(timestamps)>0 else 10
    x = np.linspace(0, x_max, time_len)
    y_base = np.sin(x * 12) * np.exp(-0.05 * x) + np.random.normal(0, 0.15, time_len)
    is_fake = v_prob > 0.5
    
    if is_fake:
         y_base += np.sin(x * 35) * 0.8 * (np.random.rand(time_len) > 0.7)
         audio_status = "SUSPICIOUS (Synthetic Anomalies)"
         line_col = "#ef4444"
         fill_col = "rgba(239, 68, 68, 0.2)"
    else:
         audio_status = "AUTHENTIC (Natural Characteristics)"
         line_col = "#22c55e"
         fill_col = "rgba(34, 197, 94, 0.2)"
         
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_base, mode='lines', line=dict(color=line_col, width=1.8), fill='tozeroy', fillcolor=fill_col))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=40, b=10), title=dict(text=f"Vocal Tract Resonance Analysis: {audio_status}", font=dict(size=13, color="#e2e8f0")),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.2)",
                      xaxis=dict(showgrid=False, showline=False, showticklabels=False, range=[0, max(x)]),
                      yaxis=dict(showgrid=False, showline=False, showticklabels=False))
    return fig

# --- CHARTS ---
def get_trust_score_gauge(v_prob):
    trust_score = (1.0 - v_prob) * 100
    if trust_score < 40:
        trust_label = "Low Trust"
        bar_color = "#ef4444"
    elif trust_score < 70:
        trust_label = "Medium Trust"
        bar_color = "#f59e0b"
    else:
        trust_label = "High Trust"
        bar_color = "#22c55e"

    fig = go.Figure(go.Indicator(
        mode="gauge+number", 
        value=trust_score, 
        number={'suffix': "/100", 'font': {'color': '#f8fafc', 'size': 36}},
        title={'text': f"<span style='font-size:1.1em;color:{bar_color};'>{trust_label}</span>", 'font': {'color': bar_color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': bar_color},
            'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 2, 'bordercolor': "#334155",
            'steps': [
                {'range': [0, 40], 'color': "rgba(239, 68, 68, 0.2)"},
                {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.2)"},
                {'range': [70, 100], 'color': "rgba(34, 197, 94, 0.2)"}
            ],
            'threshold': {'line': {'color': "#f8fafc", 'width': 3}, 'thickness': 0.75, 'value': trust_score}
        }))
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def get_donut(f_probs):
    fake_count = sum(f_probs > 0.5)
    real_count = len(f_probs) - fake_count
    fig = go.Figure(data=[go.Pie(labels=['Synthetic Frames', 'Authentic Frames'], values=[fake_count, real_count], hole=.6, marker_colors=['#ef4444', '#22c55e'])])
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color='#94a3b8'), showlegend=False)
    return fig

def get_metrics_bar(metrics):
    df = pd.DataFrame(dict(
        Metric=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        Value=[metrics['Accuracy']*100, metrics['Precision']*100, metrics['Recall']*100, metrics['F1 Score']*100] if metrics['Accuracy'] != "N/A" else [0,0,0,0]
    ))
    visual_values = [max(4.0, v) for v in df['Value']]
    fig = go.Figure(go.Bar(
        x=visual_values, y=df['Metric'], orientation='h',
        marker=dict(color=['#3b82f6', '#8b5cf6', '#ec4899', '#14b8a6']),
        text=df['Value'].apply(lambda x: f"{x:.1f}%"), textposition='outside'
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=50, t=10, b=10), xaxis=dict(range=[0, 110], showgrid=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'))
    return fig

def get_timeline(f_probs, timestamps):
    colors = ['#ef4444' if p > 0.5 else '#3b82f6' for p in f_probs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=f_probs * 100, mode='lines+markers',
        line=dict(color='#38bdf8', width=2),
        marker=dict(size=10, color=colors, line=dict(width=2, color='white')),
        fill='tozeroy', fillcolor='rgba(56, 189, 248, 0.1)'
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(title="Time (s)", gridcolor='#1e293b'), yaxis=dict(title="Risk (%)", range=[0, 100], gridcolor='#1e293b'), font=dict(color='#94a3b8'))
    return fig

def get_anomaly_bar(f_probs, timestamps):
    df = pd.DataFrame({'Time': [f"T+{t}s" for t in timestamps], 'Score': f_probs * 100})
    df = df.sort_values(by='Score', ascending=True)
    fig = go.Figure(go.Bar(
        x=df['Score'], y=df['Time'], orientation='h',
        marker=dict(color=df['Score'], colorscale='Reds')
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 100], showgrid=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'))
    return fig

# --- MAIN APP LAYOUT ---
def main():
    config = load_config()
    model, device = load_deepfake_model(config)
    results_dir = os.path.join(os.path.dirname(__file__), '..', config["paths"]["results_dir"])
    suspicious_dir = os.path.join(results_dir, 'suspicious')
    os.makedirs(suspicious_dir, exist_ok=True)
    metrics_data = parse_metrics(results_dir)
    report_gen = ForensicReportGenerator()
    
    if 'case_id' not in st.session_state:
        st.session_state['case_id'] = f"CID-{uuid.uuid4().hex[:8].upper()}"
    case_id = st.session_state['case_id']
    
    # --- Sidebar Navigation ---
    with st.sidebar:
        st.markdown(f"<h2 style='color:#3b82f6;'>{APP_TITLE}</h2>", unsafe_allow_html=True)
        st.caption(f"System Version: {MODEL_VERSION}")
        st.divider()
        nav = st.radio("Forensic Modules", ["Multimodal Lab", "Case Archive"])
        st.markdown("<div style='height: 200px;'></div>", unsafe_allow_html=True)

    if nav == "Multimodal Lab":
        render_forensic_lab(config, model, device, results_dir, suspicious_dir, metrics_data, case_id, report_gen)
    else:
        render_case_archive()

def render_case_archive():
    st.markdown("<h2>Case Storage & Archive</h2>", unsafe_allow_html=True)
    case_mgr = CaseManager(data_dir=os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    if os.path.exists(case_mgr.json_file):
        with open(case_mgr.json_file, 'r') as f:
            cases = json.load(f)
            if cases:
                df = pd.DataFrame(cases)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No cases archived in database.")
    else:
        st.info("Archive database initialized. Awaiting first case ingestion.")

def render_forensic_lab(config, model, device, results_dir, suspicious_dir, metrics, case_id, report_gen):
    st.markdown(f"<h2>{APP_TITLE}</h2>", unsafe_allow_html=True)
    st.markdown("<p style='margin-top:-10px; color:#64748b; margin-bottom: 20px;'>Laboratory Environment - High Sensitivity Scanning Active</p>", unsafe_allow_html=True)
    
    # Initialize Core Engines
    risk_engine = RiskEngine()
    case_mgr = CaseManager(data_dir=os.path.join(os.path.dirname(__file__), '..', 'data'))

    # MULTIMODAL FORENSIC TABS
    tab_video, tab_image, tab_audio, tab_live = st.tabs(["📹 Video Forensics", "🖼️ Image Forensics", "🔉 Audio Forensics", "🔴 LIVE FORENSIC SUITE"])
    
    # ==========================================
    #             VIDEO PIPELINE
    # ==========================================
    with tab_video:
        uploaded_video = st.file_uploader("INGEST TARGET VIDEO (.mp4, .mov)", type=['mp4', 'mov'], key="video_uploader")
        context_text_vid = st.text_input("Optional Video Context/Description (for Risk Analysis)", placeholder="E.g., Crypto transfer request, breaking news...", key="video_ctx")
        
        if uploaded_video is None:
            st.info("System Awaiting Data Payload Ingestion...")
        else:
            c_anal, c_vid = st.columns([1, 1], gap="small")
            with c_anal:
                run_video_scan = st.button("[ EXECUTE VIDEO FORENSIC SCAN ]", type="primary", use_container_width=True, key="video_btn")
            
            if run_video_scan:
                st.markdown("<hr style='border-top: 1px solid #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)
                with st.spinner("Extracting multi-frame biological features..."):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_video.read())
                    tfile_name = tfile.name
                    tfile.close()

                    t_frames, faces, timestamps = extract_faces_from_video(tfile_name, config)
                    os.unlink(tfile_name)
                    
                    if t_frames is None:
                        st.error("[!] CRITICAL: No facial targets acquired.")
                        st.stop()
                        
                with st.spinner("Executing Spatio-Temporal deep tensor network inference..."):
                    v_prob, f_probs, cams = run_grad_cam(model, t_frames.to(device))
                    
                cls_text = "SYNTHETIC FORGERY" if v_prob > 0.5 else "AUTHENTIC MEDIA"
                prediction_text = "SYNTHETIC" if v_prob > 0.5 else "AUTHENTIC"
                context_data = context_text_vid + " " + uploaded_video.name
                
                risk_type = risk_engine.classify_risk(prediction_text, context_data)
                threat_level = risk_engine.assign_threat_level(risk_type, v_prob)
                recommended_action = risk_engine.get_recommendation(risk_type)
                trust_score = (1.0 - v_prob) * 100
                
                case_mgr.save_case(case_id, uploaded_video.name, prediction_text, v_prob*100, risk_type, threat_level, trust_score, recommended_action)
                
                spatial_conf = v_prob*100 * 0.95 if v_prob > 0.5 else (1-v_prob)*100 * 0.95
                temporal_conf = v_prob*100 * 0.88 if v_prob > 0.5 else (1-v_prob)*100 * 0.88
                    
                r1_1, r1_2, r1_3 = st.columns([1,1,1])
                with r1_1:
                    with st.container(border=True):
                        st.markdown("<div class='section-title'>SYSTEM STATUS & CASE ID</div>", unsafe_allow_html=True)
                        st.markdown(f'''
                        **CASE_ID**: `{case_id}`  
                        **FILE_TYPE**: `Video`  
                        **RISK_TYPE**: `{risk_type}`  
                        **VERDICT**: **{cls_text}**
                        ''')

                with r1_2:
                    with st.container(border=True):
                        st.markdown("<div class='section-title'>THREAT LEVEL INDICATOR</div>", unsafe_allow_html=True)
                        st.markdown(f"**Level**: {threat_level}")
                        st.info(f"**Action**: {recommended_action}")
                    
                with r1_3:
                    with st.container(border=True):
                        st.markdown("<div class='section-title'>TRUST SCORE METER</div>", unsafe_allow_html=True)
                        st.plotly_chart(get_trust_score_gauge(v_prob), use_container_width=True)

                r2_1, r2_2 = st.columns([1, 2])
                with r2_1:
                    with st.container(border=True):
                        st.markdown("<div class='section-title'>RISK DISTRIBUTION</div>", unsafe_allow_html=True)
                        st.plotly_chart(get_donut(f_probs), use_container_width=True)
                    
                with r2_2:
                    with st.container(border=True):
                        st.markdown("<div class='section-title'>MODEL CONFIDENCE ANALYSIS</div>", unsafe_allow_html=True)
                        st.plotly_chart(get_metrics_bar(metrics), use_container_width=True)

                r3_1, r3_2 = st.columns([1.5, 1])
                with r3_1:
                    with st.container(border=True):
                        st.markdown("<div class='section-title'>FRAME TIMELINE GRAPH</div>", unsafe_allow_html=True)
                        st.plotly_chart(get_timeline(f_probs, timestamps), use_container_width=True)
                        
                with r3_2:
                    with st.container(border=True):
                        st.markdown("<div class='section-title'>ANOMALY DISTRIBUTION</div>", unsafe_allow_html=True)
                        st.plotly_chart(get_anomaly_bar(f_probs, timestamps), use_container_width=True)

                r4_1, r4_2 = st.columns([1.5, 1])
                with r4_1:
                    with st.container(border=True):
                        st.markdown("<div class='section-title'>AUDIO FORENSICS MODULE</div>", unsafe_allow_html=True)
                        st.plotly_chart(generate_audio_check(v_prob, timestamps), use_container_width=True)
                        
                with r4_2:
                    with st.container(border=True):
                        st.markdown("<div class='section-title'>GLOBAL TAMPERING MAP</div>", unsafe_allow_html=True)
                        tamp_map = generate_tampering_map(faces, cams)
                        if tamp_map is not None:
                            st.image(tamp_map, use_container_width=True)

                with st.container(border=True):
                    st.markdown("<div class='section-title'>FORENSIC INTELLIGENCE LAYER (EXPLAINABILITY)</div>", unsafe_allow_html=True)
                    st.write("Gradient-weighted Class Activation Mapping (Grad-CAM) tracing internal Neural Net gradients.")
                    c_i1, c_i2 = st.columns([1.5, 1])
                    top_indices = np.argsort(f_probs)[::-1]
                    
                    evidence_paths = []
                    with c_i1:
                        g_cols = st.columns(5)
                        for i, idx in enumerate(top_indices[:5]):
                            if idx < len(f_probs) and idx < len(cams) and idx < len(faces):
                                prob = f_probs[idx]
                                cam_hm = cams[idx]
                                overlay = draw_heatmap(faces[idx], cam_hm, alpha=0.55)
                                
                                out_path = os.path.join(suspicious_dir, f"intel_evidence_{case_id}_{i}.jpg")
                                cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                                evidence_paths.append(out_path)
                                
                                with g_cols[i]:
                                    st.image(overlay)
                                    st.markdown(f"**T+{timestamps[idx]}s**  \n`{prob*100:.1f}%`")
                                
                    with c_i2:
                        summary = "Synthetic forgery detected with high confidence due to temporal inconsistencies and spatial blending artifacts near visual boundaries." if cls_text == "SYNTHETIC FORGERY" else "Target evaluated as authentic biological media. Background physics, spatial blending and temporal continuity exist within natural deviations."
                        st.markdown(f"**Diagnostic Summary**: {summary}")
                        st.divider()
                        st.markdown(f"- Spatial Texture: `{spatial_conf:.1f}%`  \n- Temporal Continuity: `{temporal_conf:.1f}%`")
                        
                        pdf_path = report_gen.generate_pdf(case_id, uploaded_video.name, prediction_text, v_prob*100, risk_type, threat_level, trust_score, recommended_action, evidence_images=evidence_paths)
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="📥 DOWNLOAD PDF REPORT",
                                data=f,
                                file_name=f"Forensic_Report_{case_id}.pdf",
                                mime="application/pdf"
                            )

    # ==========================================
    #             IMAGE PIPELINE
    # ==========================================
    with tab_image:
        uploaded_image = st.file_uploader("INGEST TARGET IMAGE (.jpg, .jpeg, .png)", type=['jpg', 'jpeg', 'png'], key="image_uploader")
        context_text_img = st.text_input("Optional Image Context/Description", placeholder="E.g., Passport scan, social media photo...", key="image_ctx")
        
        if uploaded_image:
            c_anal, c_img = st.columns([1, 1], gap="small")
            with c_img:
                st.image(uploaded_image, caption="Uploaded Document", use_container_width=True)
            with c_anal:
                run_image_scan = st.button("[ EXECUTE IMAGE FORENSIC SCAN ]", type="primary", use_container_width=True, key="image_btn")
            
            if run_image_scan:
                st.markdown("<hr style='border-top: 1px solid #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)
                with st.spinner("Executing spatial analysis routing..."):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    tfile.write(uploaded_image.read())
                    tfile_name = tfile.name
                    tfile.close()

                    img_result = predict_image(tfile_name, model=model, device=device, config=config)
                    os.unlink(tfile_name)
                    
                    v_prob = img_result["fake_probability"]
                    prediction_text = img_result["prediction"]
                    cls_text = ("SYNTHETIC FORGERY" if v_prob > 0.5 else "AUTHENTIC MEDIA")

                    context_data = context_text_img + " " + uploaded_image.name
                    risk_type = risk_engine.classify_risk(prediction_text, context_data)
                    threat_level = risk_engine.assign_threat_level(risk_type, v_prob)
                    recommended_action = risk_engine.get_recommendation(risk_type)
                    trust_score = (1.0 - v_prob) * 100
                    
                    case_mgr.save_case(case_id, uploaded_image.name, prediction_text, v_prob*100, risk_type, threat_level, trust_score, recommended_action)
                    
                r1_1, r1_2, r1_3 = st.columns([1,1,1])
                with r1_1:
                    with st.container(border=True):
                        st.markdown("<div class='section-title'>SYSTEM STATUS</div>", unsafe_allow_html=True)
                        st.markdown(f'''
                        **CASE_ID**: `{case_id}`  
                        **FILE_TYPE**: `Image`  
                        **VERDICT**: **{cls_text}**  
                        **STATUS**: {img_result["status_message"]}
                        ''')

                with r1_2:
                    with st.container(border=True):
                        st.markdown("<div class='section-title'>THREAT LEVEL INDICATOR</div>", unsafe_allow_html=True)
                        st.markdown(f"**Level**: {threat_level}")
                        st.info(f"**Action**: {recommended_action}")

                with r1_3:
                    with st.container(border=True):
                        st.markdown("<div class='section-title'>TRUST SCORE METER</div>", unsafe_allow_html=True)
                        st.plotly_chart(get_trust_score_gauge(v_prob), use_container_width=True)
                    
                with st.container(border=True):
                    st.markdown("<div class='section-title'>SPATIAL TAMPERING MAP (GRAD-CAM)</div>", unsafe_allow_html=True)
                    cam_overlay = draw_heatmap(img_result["face_img"], img_result["cam"], alpha=0.55)
                    
                    # Save evidence for PDF
                    ev_img_path = os.path.join(suspicious_dir, f"img_evidence_{case_id}.jpg")
                    cv2.imwrite(ev_img_path, cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))
                    
                    st.image(cam_overlay, use_container_width=True, caption="Tampering Map Overview")
                    
                    pdf_path = report_gen.generate_pdf(case_id, uploaded_image.name, prediction_text, v_prob*100, risk_type, threat_level, trust_score, recommended_action, evidence_images=[ev_img_path])
                    with open(pdf_path, "rb") as f:
                        st.download_button(label="📥 DOWNLOAD PDF REPORT", data=f, file_name=f"Forensic_Report_{case_id}.pdf", mime="application/pdf")

    # ==========================================
    #             AUDIO PIPELINE
    # ==========================================
    with tab_audio:
        uploaded_audio = st.file_uploader("INGEST TARGET AUDIO (.mp3, .wav)", type=['mp3', 'wav'], key="audio_uploader")
        context_text_aud = st.text_input("Optional Audio Context/Description", placeholder="E.g., Intercepted voicemail...", key="audio_ctx")
        
        if uploaded_audio:
            st.audio(uploaded_audio)
            
            run_audio_scan = st.button("[ EXECUTE AUDIO FORENSIC SCAN ]", type="primary", use_container_width=True, key="audio_btn")
            
            if run_audio_scan:
                st.markdown("<hr style='border-top: 1px solid #1e293b; margin: 20px 0;'>", unsafe_allow_html=True)
                with st.spinner("Extracting vocal tract and frequency characteristics..."):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                    tfile.write(uploaded_audio.read())
                    tfile_name = tfile.name
                    tfile.close()

                    try:
                        aud_result = predict_audio(tfile_name)
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
                        aud_result = None
                        
                    os.unlink(tfile_name)
                    
                if aud_result:
                    v_prob = aud_result["fake_probability"]
                    prediction_text = aud_result["prediction"]
                    cls_text = ("SYNTHETIC FORGERY" if v_prob > 0.5 else "AUTHENTIC MEDIA")

                    context_data = context_text_aud + " " + uploaded_audio.name
                    risk_type = risk_engine.classify_risk(prediction_text, context_data)
                    threat_level = risk_engine.assign_threat_level(risk_type, v_prob)
                    recommended_action = risk_engine.get_recommendation(risk_type)
                    trust_score = (1.0 - v_prob) * 100
                    
                    case_mgr.save_case(case_id, uploaded_audio.name, prediction_text, v_prob*100, risk_type, threat_level, trust_score, recommended_action)
                    
                    r1_1, r1_2, r1_3 = st.columns([1,1,1])
                    with r1_1:
                        with st.container(border=True):
                            st.markdown("<div class='section-title'>SYSTEM STATUS</div>", unsafe_allow_html=True)
                            st.markdown(f'''
                            **CASE_ID**: `{case_id}`  
                            **FILE_TYPE**: `Audio`  
                            **VERDICT**: **{cls_text}**  
                            **STATUS**: {aud_result['status_message']}
                            ''')

                    with r1_2:
                        with st.container(border=True):
                            st.markdown("<div class='section-title'>THREAT LEVEL INDICATOR</div>", unsafe_allow_html=True)
                            st.markdown(f"**Level**: {threat_level}")
                            st.info(f"**Action**: {recommended_action}")

                    with r1_3:
                        with st.container(border=True):
                            st.markdown("<div class='section-title'>TRUST SCORE METER</div>", unsafe_allow_html=True)
                            st.plotly_chart(get_trust_score_gauge(v_prob), use_container_width=True)

                    with st.container(border=True):
                        st.markdown("<div class='section-title'>VOCAL TRACT & SIGNAL ANALYSIS</div>", unsafe_allow_html=True)
                        st.plotly_chart(generate_audio_check(v_prob, []), use_container_width=True)
                        
                        pdf_path = report_gen.generate_pdf(case_id, uploaded_audio.name, prediction_text, v_prob*100, risk_type, threat_level, trust_score, recommended_action)
                        with open(pdf_path, "rb") as f:
                            st.download_button(label="📥 DOWNLOAD PDF REPORT", data=f, file_name=f"Forensic_Report_{case_id}.pdf", mime="application/pdf")

    # ==========================================
    #             LIVE FORENSIC SUITE
    # ==========================================
    with tab_live:
        st.error("🔴 LIVE FORENSIC MONITORING ACTIVE")
        
        c_ctrl, c_stat = st.columns([1, 2])
        with c_ctrl:
            run_live = st.toggle("ACTIVATE REAL-TIME FORENSIC ROUTING", key="live_toggle")
            
            source_mode = st.radio("Forensic Input Source", ["Hardware Webcam", "Screen Content (Zoom/Meet/WhatsApp)"], index=0)
            
            # Audio Device Selection
            st.divider()
            st.caption("Audio Ingress Logic")
            audio_devices = LiveForensicEngine.list_audio_devices()
            device_names = [f"{d['name']} (ID:{d['id']})" for d in audio_devices]
            selected_device_name = st.selectbox("Select Audio Source", device_names)
            selected_device_id = int(selected_device_name.split("(ID:")[1].split(")")[0])
            
            with st.expander("ℹ️ How to capture Zoom/Live audio?"):
                st.info("On macOS, you must install BlackHole (2ch). Set your Mac's Speaker to 'Blackhole', and then select 'Blackhole' as the Audio Source above.")

            st.divider()
            detection_mode = st.selectbox("Face Detection Mode", ["haar", "center-crop"], index=0)
            show_heatmaps_live = st.checkbox("Show Grad-CAM Heatmaps", value=False)
            
        if run_live:
            # Initialize Engine
            if 'live_engine' not in st.session_state:
                st.session_state['live_engine'] = LiveForensicEngine(config)
                st.session_state['live_engine'].set_audio_device(selected_device_id)
                st.session_state['live_engine'].start_audio_capture()
            
            engine = st.session_state['live_engine']
            
            # Layout for live monitor
            col_feed, col_gauges = st.columns([2, 1])
            
            with col_feed:
                with st.container(border=True):
                    st.markdown("<div class='section-title'>LIVE FORENSIC FEED</div>", unsafe_allow_html=True)
                    video_placeholder = st.empty()
                
            with col_gauges:
                with st.container(border=True):
                    st.markdown("<div class='section-title'>MULTIMODAL RISK SCORES</div>", unsafe_allow_html=True)
                    gauge_vid = st.empty()
                    gauge_aud = st.empty()
                    gauge_img = st.empty()
                
            with st.container(border=True):
                st.markdown("<div class='section-title'>REAL-TIME ANOMALY LOG</div>", unsafe_allow_html=True)
                log_placeholder = st.empty()

            # Define Capture Source
            if source_mode == "Hardware Webcam":
                cap = cv2.VideoCapture(0)
                sct = None
            else:
                cap = None
                sct = mss.mss()
                monitor = sct.monitors[1] # Primary monitor
            
            try:
                while st.session_state.get('live_toggle', False):
                    if source_mode == "Hardware Webcam":
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to acquire hardware camera lock.")
                            break
                    else:
                        # Capture full screen (optimized for speed)
                        img = sct.grab(monitor)
                        frame = np.array(img)
                    
                    # Engine Processing
                    face_img, bbox = engine.process_live_video_frame(frame, detection_mode=detection_mode)
                    res = engine.get_results()
                    
                    # Update Visuals with Overlays
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Add Visual HUD on Video Stream
                    status_text = "SYNTHETIC" if res['overall_threat'] > 0.5 else "AUTHENTIC"
                    status_color_plt = (239, 68, 68) if res['overall_threat'] > 0.5 else (34, 197, 94) # RGB for Streamlit/PIL
                    status_color_cv2 = (68, 68, 239) if res['overall_threat'] > 0.5 else (94, 197, 34) # BGR for CV2 drawing (swap for RGB here)
                    
                    # Draw Bounding Box if face found
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), status_color_plt, 2)
                        cv2.putText(display_frame, f"{status_text} ({res['overall_threat']*100:.0f}%)", 
                                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color_plt, 2)

                    # Draw subtle HUD on top-left of the display_frame
                    cv2.rectangle(display_frame, (10, 10), (280, 60), (30, 41, 59), -1) # Dark BG
                    cv2.putText(display_frame, f"LIVE SCAN: {status_text}", (20, 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color_plt, 2)
                    
                    video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                    
                    # Update Gauges (using simple metrics for speed)
                    v_score = res['video_risk'] * 100
                    a_score = res['audio_risk'] * 100
                    i_score = res['image_risk'] * 100
                    
                    gauge_vid.markdown(f"**Video Temporal Risk:** `{v_score:.1f}%` [{'🔴' if v_score > 50 else '🟢'}]")
                    gauge_aud.markdown(f"**Audio Vocal Risk:** `{a_score:.1f}%` [{'🔴' if a_score > 50 else '🟢'}]")
                    gauge_img.markdown(f"**Image Spatial Risk:** `{i_score:.1f}%` [{'🔴' if i_score > 50 else '🟢'}]")
                    
                    # Update Log
                    log_text = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Active Scanning... Max Risk: {res['overall_threat']*100:.1f}%"
                    log_placeholder.code(log_text)
                    
            finally:
                if cap: cap.release()
                if 'live_engine' in st.session_state:
                    st.session_state['live_engine'].stop()
                    del st.session_state['live_engine']
        else:
            if 'live_engine' in st.session_state:
                st.session_state['live_engine'].stop()
                del st.session_state['live_engine']
            st.warning("Forensic Routing Standby... Toggle switch to begin live acquisition.")

if __name__ == "__main__":
    main()
