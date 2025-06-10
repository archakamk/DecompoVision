'''import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models.model import DecompoVisionModel

image_path = "room.jpg"
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

input_tensor = transform(image).unsqueeze(0)

model = DecompoVisionModel()
model.eval()

with torch.no_grad():
    albedo, shading, shadow, specular = model(input_tensor)

def show_tensor(tensor, title):
    img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)  # Normalize for display
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

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
plt.show()'''


import matplotlib.pyplot as plt
from PIL import Image
from src.inference import run_inference

def show_output(tensor, title, idx):
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min() + 1e-5)
    plt.subplot(2, 3, idx)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

if __name__ == "__main__":
    image_path = "indoor_room.jpg"  # Replace with your image
    original_image = Image.open(image_path).convert("RGB")

    albedo, shading, shadow, specular = run_inference(image_path)

    plt.figure(figsize=(12, 8))

    # Display original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')

    # Display model outputs
    show_output(albedo, "Albedo", 2)
    show_output(shading, "Shading", 3)
    show_output(shadow, "Shadow", 4)
    show_output(specular, "Specular", 5)

    plt.tight_layout()
    plt.show()

