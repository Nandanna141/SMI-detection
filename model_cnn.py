import torch
import torch.nn as nn

class ECG1DCNN(nn.Module):
    def __init__(self):
        super(ECG1DCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fully Connected Classifier Layers
        # Assuming an input heartbeat window size of 256 samples
        self.fc1 = nn.Linear(32 * 64, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Prevents overfitting
        self.fc2 = nn.Linear(64, 2) # Binary Classification: 0 = Normal, 1 = Ischemic/SMI Risk

    def forward(self, x):
        # Input shape: (Batch Size, Channels=1, Sequence Length=256)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Flatten for linear layers
        x = x.view(x.size(0), -1)
        
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

print("--- Testing 1D-CNN Model Initialization ---")
model = ECG1DCNN()
print(model)

# Create a dummy batch of heartbeats to test a forward pass (Batch size = 4, Channels = 1, Length = 256)
dummy_input = torch.randn(4, 1, 256)
output = model(dummy_input)
print("\nForward pass successful!")
print("Output logits shape for batch:", output.shape)