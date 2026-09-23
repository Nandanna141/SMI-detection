import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib
matplotlib.use('Agg') # Prevents any GUI window freezing issues
import matplotlib.pyplot as plt

print("--- Step 1: Simulating Raw Wearable ECG Data ---")
raw_ecg = nk.ecg_simulate(duration=10, sampling_rate=250, noise=0.2)

print("--- Step 2: Applying Bandpass Filtering & Cleaning ---")
cleaned_ecg = nk.ecg_clean(raw_ecg, sampling_rate=250, method='neurokit')

print("--- Step 3: Detecting R-Peaks (Heartbeats) ---")
_, signals = nk.ecg_peaks(cleaned_ecg, sampling_rate=250)
rpeaks = signals["ECG_R_Peaks"]
detected_beats = np.sum(rpeaks == 1)
print(f"-> Successfully detected {detected_beats} heartbeats in the 10-second window.")

print("--- Step 4: Saving the Result Plot ---")
plt.figure(figsize=(10, 4))
plt.plot(raw_ecg[:500], label="Raw Noisy ECG", color="red", alpha=0.6)
plt.plot(cleaned_ecg[:500], label="Cleaned ECG", color="green", linewidth=2)
plt.title("ECG Preprocessing Pipeline (First 2 Seconds)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.savefig("ecg_plot.png")
print("-> Plot saved successfully as 'ecg_plot.png' in your folder!")