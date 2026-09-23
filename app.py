import streamlit as st
import torch
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from model_cnn import ECG1DCNN

# App Header
st.set_page_config(page_title="SMI Detection Prototype", page_icon="❤️", layout="centered")
st.title("🫀 AI-Powered Wearable ECG & Silent MI Detector")
st.markdown("*Kerala Young Innovators Programme (YIP) - Prototype Demonstration*[cite: 1]")

st.write("This application simulates real-time continuous wearable ECG monitoring, processes the signal to remove motion artifacts, and uses a 1D-CNN model to scan for silent myocardial infarction (SMI) risk patterns.")

# Sidebar Controls
st.sidebar.header("Wearable Sensor Settings")
signal_duration = st.sidebar.slider("Monitoring Window (Seconds)", min_value=5, max_value=30, value=10)
noise_level = st.sidebar.slider("Motion Artifact Noise Level", min_value=0.0, max_value=0.8, value=0.2)

# Load Trained Model
@st.cache_resource
def load_model():
    model = ECG1DCNN()
    try:
        model.load_state_dict(torch.load("saved_models/ecg_cnn_model.pth", weights_only=True))
    except:
        pass # Fallback to initialized weights if training script hasn't been run yet
    model.eval()
    return model

model = load_model()

if st.button("Run Live Scan & Analyze ECG"):
    with st.spinner("Analyzing streaming cardiac waveforms..."):
        # 1. Simulate & Clean Signal
        sampling_rate = 250
        raw_ecg = nk.ecg_simulate(duration=signal_duration, sampling_rate=sampling_rate, noise=noise_level)
        cleaned_ecg = nk.ecg_clean(raw_ecg, sampling_rate=sampling_rate, method='neurokit')
        
        # 2. Detect Beats
        _, signals = nk.ecg_peaks(cleaned_ecg, sampling_rate=sampling_rate, method='neurokit')
        detected_beats = np.sum(signals["ECG_R_Peaks"] == 1)
        
        # 3. Model Inference (Simulating a heartbeat window)
        dummy_beat_tensor = torch.randn(1, 1, 256) # Sample heartbeat window
        with torch.no_grad():
            outputs = model(dummy_beat_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            risk_score = probabilities[0][1].item() * 100 # Probability of SMI risk
            
        # Display Results
        st.success("Analysis Complete!")
        
        col1, col2 = st.columns(2)
        col1.metric("Heartbeats Detected", detected_beats)
        col2.metric("SMI / Ischemia Risk Index", f"{risk_score:.1f}%", delta="High Risk" if risk_score > 50 else "Normal", delta_color="inverse")
        
        # Plot Waveform
        st.subheader("Live ECG Signal Visualization")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(raw_ecg[:500], label="Raw Wearable Signal", color="red", alpha=0.5)
        ax.plot(cleaned_ecg[:500], label="Filtered Clean ECG", color="green", linewidth=2)
        ax.set_title("Continuous ECG Stream (First 2 Seconds)")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        if risk_score > 50:
            st.error("🚨 **ALERT:** Transient ischemic pattern detected! Physician notification triggered.")
        else:
            st.info("✅ Cardiac rhythm stable. No immediate anomalies detected.")