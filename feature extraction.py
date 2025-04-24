import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset_folder = r"C:\Users\naman\OneDrive\Desktop\Tumor"

image_files = [f for f in os.listdir(dataset_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

for image_file in image_files:
    img_path = os.path.join(dataset_folder, image_file)
    img = Image.open(img_path).convert("RGB")

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        features = model(img_tensor)

    feature_vector = features.squeeze().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(img)
    ax[0].axis("off")
    ax[0].set_title(f"Original MRI: {image_file}")

    ax[1].imshow(feature_vector.reshape(16, -1), cmap="viridis", aspect="auto")
    ax[1].axis("off")
    ax[1].set_title("Extracted Features (SSL)")

    plt.tight_layout()
    plt.show()

