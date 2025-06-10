import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from models.model import DecompoVisionModel

# Load and preprocess input image
image_path = "room.jpg"  # Change this to your test image path
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

input_tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]

# Initialize model
model = DecompoVisionModel()
model.eval()

# Run inference
with torch.no_grad():
    albedo, shading, shadow, specular = model(input_tensor)

# Function to display tensor output
def show_tensor(tensor, title):
    img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)  # Normalize for display
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

# Visualize results
plt.figure(figsize=(10, 8))

plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title("Input Image")
plt.axis("off")

plt.subplot(2, 3, 2)
show_tensor(albedo, "Albedo")

plt.subplot(2, 3, 3)
show_tensor(shading, "Shading")

plt.subplot(2, 3, 4)
show_tensor(shadow, "Shadow")

plt.subplot(2, 3, 5)
show_tensor(specular, "Specular")

plt.tight_layout()
plt.show()
