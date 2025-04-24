import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# ==============================
# 🔹 Dataset Class
# ==============================
class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

        if not self.images:
            raise FileNotFoundError(f"No images found in {root_dir}. Check dataset location!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")  # Ensure 3-channel image

        if self.transform:
            image = self.transform(image)

        return image

# ==============================
# 🔹 Transformations
# ==============================
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# ==============================
# 🔹 Load Dataset
# ==============================
dataset_path = "C:/Users/naman/OneDrive/Desktop/python project/braintumor"
dataset = MRIDataset(dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ==============================
# 🔹 Simple Self-Supervised Model
# ==============================
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

# ==============================
# 🔹 Train Loop
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleSSLModel().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Dummy loss for SSL (modify as needed)

# Dummy target for SSL (identity function)
def generate_ssl_target(batch):
    return batch.clone().detach()

# Train for 5 epochs
for epoch in range(5):
    for images in dataloader:
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, generate_ssl_target(outputs))  # Dummy target
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/5] - Loss: {loss.item():.4f}")

print("✅ Training complete!")
