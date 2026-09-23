import torch
import pandas as pd
import neurokit2 as nk
import numpy as np

print("--- Environment Test ---")
print("PyTorch Version:", torch.__version__)

# Generate a synthetic ECG signal using NeuroKit2 to test signal processing
ecg_signal = nk.ecg_simulate(duration=10, sampling_rate=250)
print("Successfully generated synthetic ECG signal of length:", len(ecg_signal))
