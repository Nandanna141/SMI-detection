import torch
import torch.nn as nn
import torch.optim as optim
from model_cnn import ECG1DCNN


print("--- Initializing Model, Loss, and Optimizer ---")
model = ECG1DCNN()
criterion = nn.CrossEntropyLoss() # Standard loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("--- Generating Synthetic Training Dataset ---")
# Let's create a synthetic dataset of 100 heartbeats (Batch, Channels, Length)
# Class 0: Normal beats, Class 1: Ischemic/SMI risk beats
X_train = torch.randn(100, 1, 256)
y_train = torch.randint(0, 2, (100,)) # Random labels 0 or 1

print("--- Starting Training Loop (3 Epochs) ---")
model.train()
for epoch in range(3):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    # Calculate training accuracy
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == y_train).sum().item()
    accuracy = (correct / y_train.size(0)) * 100
    
    print(f"Epoch [{epoch+1}/3] | Loss: {loss.item():.4f} | Training Accuracy: {accuracy:.2f}%")

print("\n--- Training Complete! Model is ready for inference testing. ---")
import os

# Save the trained model weights
os.makedirs("saved_models", exist_ok=True)
model_path = "saved_models/ecg_cnn_model.pth"
torch.save(model.state_dict(), model_path)
print(f"\n--- Model successfully saved to {model_path} ---")