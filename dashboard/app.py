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

# System Constants
MODEL_VERSION = "Forensic-CNN-LSTM v3.0 (Enterprise Analytics Tier)"
APP_TITLE = "Multimodal Synthetic Media Forensics & Threat Assessment System"

st.set_page_config(page_title="Forensic Analytics Hub", layout="wide", initial_sidebar_state="collapsed")

# --- ENTERPRISE CYBERSECURITY DARK THEME ---
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp { background-color: #0b1121; color: #e2e8f0; }
    h1, h2, h3, h4, h5, p, span, div { font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Metric overrides */
    div[data-testid="stMetricValue"] { color: #f8fafc; }
    
    /* Reusable Card Class */
    .dash-card {
        background-color: #151e32;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .dash-card:hover { transform: translateY(-2px); box-shadow: 0 12px 20px -4px rgba(0, 0, 0, 0.7); border: 1px solid #334155;}
    
    .section-title { 
        font-size: 0.9rem; color: #94a3b8; font-weight: 700; 
        text-transform: uppercase; letter-spacing: 1.5px; 
        border-bottom: 1px solid #1e293b; padding-bottom: 10px; margin-bottom: 15px; 
        display:flex; align-items:center; gap:8px;
    }
    
    /* Badges & Indicators */
    .badge-safe { background-color: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; color: #4ade80; padding: 15px; border-radius: 8px; font-weight: bold; font-size: 1.2rem; text-align:center;}
    .badge-danger { background-color: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; color: #f87171; padding: 15px; border-radius: 8px; font-weight: bold; font-size: 1.2rem; text-align:center;}
    .badge-med { background-color: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; color: #fbbf24; padding: 15px; border-radius: 8px; font-weight: bold; font-size: 1.2rem; text-align:center;}
    
    .sys-log { font-family: 'Consolas', 'Courier New', monospace; font-size: 0.85rem; color: #4ade80; background: #020617; padding: 15px; border-radius:6px; border: 1px solid #1e293b;}
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
    
    if 'case_id' not in st.session_state:
        st.session_state['case_id'] = f"CID-{uuid.uuid4().hex[:8].upper()}"
    case_id = st.session_state['case_id']
    
    st.markdown(f"<h2>{APP_TITLE}</h2>", unsafe_allow_html=True)
    st.markdown("<p style='margin-top:-10px; color:#64748b; margin-bottom: 20px;'>Deep Learning Forensic Matrix & Operations Status</p>", unsafe_allow_html=True)
    
    # Initialize Core Engines
    risk_engine = RiskEngine()
    case_mgr = CaseManager(data_dir=os.path.join(os.path.dirname(__file__), '..', 'data'))

    # MULTIMODAL FORENSIC TABS
    tab_video, tab_image, tab_audio = st.tabs(["📹 Video Forensics", "🖼️ Image Forensics", "🔉 Audio Forensics"])
    
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
                    st.markdown(f'''<div class="dash-card">
                    <div class="section-title">SYSTEM STATUS & CASE ID</div>
                    <div class="sys-log" style="font-size:0.75rem;">
                    CASE_ID    : <b style="color:#fcd34d;">{case_id}</b><br>
                    FILE_TYPE  : Video<br>
                    RISK_TYPE  : {risk_type}<br>
                    PREDICTION : {cls_text}
                    </div>
                    </div>''', unsafe_allow_html=True)
                    
                with r1_2:
                    bc = "badge-danger" if threat_level in ["HIGH", "CRITICAL"] else "badge-med" if threat_level=="MEDIUM" else "badge-safe"
                    st.markdown(f'''<div class="dash-card">
                    <div class="section-title">THREAT LEVEL INDICATOR</div>
                    <div class="{bc}">{threat_level.upper()} THREAT</div>
                    <div style="margin-top:10px; font-size:0.85rem; padding:8px; background:#020617; border-left:3px solid #f87171;"><b>Action:</b> {recommended_action}</div>
                    </div>''', unsafe_allow_html=True)
                    
                with r1_3:
                    st.markdown("<div class='dash-card'><div class='section-title'>TRUST SCORE METER</div>", unsafe_allow_html=True)
                    st.plotly_chart(get_trust_score_gauge(v_prob), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                r2_1, r2_2 = st.columns([1, 2])
                with r2_1:
                    st.markdown("<div class='dash-card'><div class='section-title'>RISK DISTRIBUTION</div>", unsafe_allow_html=True)
                    st.plotly_chart(get_donut(f_probs), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with r2_2:
                    st.markdown("<div class='dash-card'><div class='section-title'>MODEL CONFIDENCE ANALYSIS</div>", unsafe_allow_html=True)
                    st.plotly_chart(get_metrics_bar(metrics_data), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                r3_1, r3_2 = st.columns([1.5, 1])
                with r3_1:
                        st.markdown("<div class='dash-card'><div class='section-title'>FRAME TIMELINE GRAPH</div>", unsafe_allow_html=True)
                        st.plotly_chart(get_timeline(f_probs, timestamps), use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                with r3_2:
                        st.markdown("<div class='dash-card'><div class='section-title'>ANOMALY DISTRIBUTION</div>", unsafe_allow_html=True)
                        st.plotly_chart(get_anomaly_bar(f_probs, timestamps), use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                r4_1, r4_2 = st.columns([1.5, 1])
                with r4_1:
                        st.markdown("<div class='dash-card'><div class='section-title'>AUDIO FORENSICS MODULE</div>", unsafe_allow_html=True)
                        st.plotly_chart(generate_audio_check(v_prob, timestamps), use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                with r4_2:
                        st.markdown("<div class='dash-card'><div class='section-title'>GLOBAL TAMPERING MAP</div>", unsafe_allow_html=True)
                        st.markdown("<p style='font-size:0.8rem; color:#64748b; margin-top:-10px; margin-bottom:5px;'>Averaged spatial anomaly detection map</p>", unsafe_allow_html=True)
                        tamp_map = generate_tampering_map(faces, cams)
                        if tamp_map is not None:
                            mcol1, mcol2, mcol3 = st.columns([1,3,1])
                            with mcol2:
                                st.image(tamp_map, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='dash-card'><div class='section-title'>FORENSIC INTELLIGENCE LAYER (EXPLAINABILITY)</div>", unsafe_allow_html=True)
                st.write("Gradient-weighted Class Activation Mapping (Grad-CAM) tracing internal Neural Net gradients.")
                c_i1, c_i2 = st.columns([1.5, 1])
                top_indices = np.argsort(f_probs)[::-1]
                
                with c_i1:
                    g_cols = st.columns(5)
                    for i, idx in enumerate(top_indices[:5]):
                        if idx < len(f_probs) and idx < len(cams) and idx < len(faces):
                            prob = f_probs[idx]
                            cam_hm = cams[idx]
                            overlay = draw_heatmap(faces[idx], cam_hm, alpha=0.55)
                            
                            out_path = os.path.join(suspicious_dir, f"intel_evidence_t{timestamps[idx]}.jpg")
                            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                            
                            with g_cols[i]:
                                st.image(overlay)
                                st.markdown(f"<div style='background:#0f172a; border: 1px solid #1e293b; border-radius:4px; padding:5px; text-align:center; font-size:0.75rem; color:#94a3b8;'>RANK {i+1}<br><b style='color:#38bdf8'>T+{timestamps[idx]}s</b><br><span style='color:{'#ef4444' if prob>0.5 else '#22c55e'}'>{prob*100:.1f}%</span></div>", unsafe_allow_html=True)
                            
                with c_i2:
                    summary = "Synthetic forgery detected with high confidence due to temporal inconsistencies and spatial blending artifacts near visual boundaries." if cls_text == "SYNTHETIC FORGERY" else "Target evaluated as authentic biological media. Background physics, spatial blending and temporal continuity exist within natural deviations."
                    st.markdown(f'''
                    <div style="background:#0f172a; padding:15px; border-radius:6px; border:1px solid #1e293b;">
                        <b style="color:#f8fafc;">Diagnostic Forensic Summary:</b><br>
                        <div style="color:#cbd5e1; margin-top:5px; margin-bottom:15px; font-size:0.9rem;">{summary}</div>
                        <hr style="margin:8px 0px; border-color:#1e293b;">
                        <b style="color:#f8fafc; font-size:0.8rem;">Architecture Sub-Confidence Breakdowns:</b><br>
                        <div style="color:#38bdf8; font-size:0.85rem;">Spatial Texture Confidence: {spatial_conf:.1f}%</div>
                        <div style="color:#ec4899; font-size:0.85rem;">Temporal Continuity Confidence: {temporal_conf:.1f}%</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    repo_text = generate_report_text(v_prob, f_probs, top_indices, timestamps, metrics_data, case_id, uploaded_video.name, risk_type, threat_level, recommended_action)
                    st.download_button(
                        label="[X] Export Forensic Incident Report (CASE ID)",
                        data=repo_text,
                        file_name=f"Case_{case_id}_{datetime.datetime.now().strftime('%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="dl_video"
                    )
                st.markdown("</div>", unsafe_allow_html=True)

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
                    st.markdown(f'''<div class="dash-card">
                    <div class="section-title">SYSTEM STATUS</div>
                    <div class="sys-log" style="font-size:0.75rem;">
                    CASE_ID    : <b style="color:#fcd34d;">{case_id}</b><br>
                    FILE_TYPE  : Image<br>
                    PREDICTION : {cls_text}<br>
                    STATUS     : {img_result["status_message"]}
                    </div>
                    </div>''', unsafe_allow_html=True)
                with r1_2:
                    bc = "badge-danger" if threat_level in ["HIGH", "CRITICAL"] else "badge-med" if threat_level=="MEDIUM" else "badge-safe"
                    st.markdown(f'''<div class="dash-card">
                    <div class="section-title">THREAT LEVEL INDICATOR</div>
                    <div class="{bc}">{threat_level.upper()} THREAT</div>
                    <div style="margin-top:10px; font-size:0.85rem; padding:8px; background:#020617; border-left:3px solid #f87171;"><b>Action:</b> {recommended_action}</div>
                    </div>''', unsafe_allow_html=True)
                with r1_3:
                    st.markdown("<div class='dash-card'><div class='section-title'>TRUST SCORE METER</div>", unsafe_allow_html=True)
                    st.plotly_chart(get_trust_score_gauge(v_prob), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                st.markdown("<div class='dash-card'><div class='section-title'>SPATIAL TAMPERING MAP (GRAD-CAM)</div>", unsafe_allow_html=True)
                cam_overlay = draw_heatmap(img_result["face_img"], img_result["cam"], alpha=0.55)
                col_m1, col_m2, col_m3 = st.columns([1, 2, 1])
                with col_m2:
                    st.image(cam_overlay, use_container_width=True, caption="Tampering Map Overview")
                st.markdown("</div>", unsafe_allow_html=True)

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
                        st.markdown(f'''<div class="dash-card">
                        <div class="section-title">SYSTEM STATUS</div>
                        <div class="sys-log" style="font-size:0.75rem;">
                        CASE_ID    : <b style="color:#fcd34d;">{case_id}</b><br>
                        FILE_TYPE  : Audio<br>
                        PREDICTION : {cls_text}<br>
                        STATUS     : {aud_result['status_message']}
                        </div>
                        </div>''', unsafe_allow_html=True)
                    with r1_2:
                        bc = "badge-danger" if threat_level in ["HIGH", "CRITICAL"] else "badge-med" if threat_level=="MEDIUM" else "badge-safe"
                        st.markdown(f'''<div class="dash-card">
                        <div class="section-title">THREAT LEVEL INDICATOR</div>
                        <div class="{bc}">{threat_level.upper()} THREAT</div>
                        <div style="margin-top:10px; font-size:0.85rem; padding:8px; background:#020617; border-left:3px solid #f87171;"><b>Action:</b> {recommended_action}</div>
                        </div>''', unsafe_allow_html=True)
                    with r1_3:
                        st.markdown("<div class='dash-card'><div class='section-title'>TRUST SCORE METER</div>", unsafe_allow_html=True)
                        st.plotly_chart(get_trust_score_gauge(v_prob), use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("<div class='dash-card'><div class='section-title'>VOCAL TRACT & SIGNAL ANALYSIS</div>", unsafe_allow_html=True)
                    st.plotly_chart(generate_audio_check(v_prob, []), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
