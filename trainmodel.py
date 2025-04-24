import torch
import torch.nn as nn

class SimpleSSLModel(nn.Module):
    def __init__(self):
        super(SimpleSSLModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * 32 * 32, 128)  # Fully connected layer

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

# Load Model
model = SimpleSSLModel()
state_dict = torch.load("trainmodel.pth", map_location="cpu")

# Try loading the state dict
try:
    model.load_state_dict(state_dict)
    print("✅ Model Loaded Successfully!")
except RuntimeError as e:
    print(f"❌ Error Loading Model: {e}")
torch.save(model.state_dict(), "trainmodel_new.pth")