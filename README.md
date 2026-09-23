🫀 AI-Powered Continuous Wearable ECG & Silent Myocardial Infarction (SMI) Detector
 
> An edge-optimized deep learning software solution for real-time cardiac monitoring and early detection of silent heart attacks.


📌 Project Overview
Existing cardiac monitoring solutions (such as traditional Holter monitors) are often bulky, expensive, and fail to provide continuous, real-time tracking for asymptomatic or **"silent" myocardial infarctions (SMI)**. This project presents an end-to-end AI prototype that combines digital signal processing with a lightweight **1D-Convolutional Neural Network (1D-CNN)** to analyze wearable ECG streams and flag ischemic risk patterns instantly.

 🚀 Key Features
* **Signal Processing Pipeline:** Automated noise filtering and motion artifact removal using `NeuroKit2`[cite: 1, 14].
* **Deep Learning Core:** Custom PyTorch 1D-CNN architecture designed to ingest heartbeat windows and classify anomalies[cite: 1, 15].
* **Interactive Web Prototype:** A live `Streamlit` dashboard providing real-time signal visualization and risk score calculation[cite: 1, 15].
* **Edge-Ready Design:** Lightweight model architecture optimized for low-power wearable microcontrollers.

 🛠️ Tech Stack
* **Language:** Python 3.13[cite: 4, 8]
* **Deep Learning:** PyTorch[cite: 1, 15]
* **Signal Processing:** NeuroKit2[cite: 1, 14]
* **UI Framework:** Streamlit[cite: 1, 15]
* **Visualization:** Matplotlib[cite: 15]

 📂 Repository Structure
text
SMI-detection/
app.py                  # Interactive Streamlit web prototype
 model_cnn.py            # PyTorch 1D-CNN model architecture
 train_model.py          # Training and evaluation script
saved_models/
ecg_cnn_model.pth   # Serialized trained model weights
 ecg_pipeline.py         # Digital signal preprocessing pipeline
 README.md               # Project documentation
