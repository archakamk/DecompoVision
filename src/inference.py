import torch
from torchvision import transforms
from PIL import Image
from models.model import DecompoVisionModel

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]

# utils/inference.py (continued)

def run_inference(image_path):
    model = DecompoVisionModel()
    model.eval()

    input_tensor = preprocess_image(image_path)

    with torch.no_grad():
        albedo, shading, shadow, specular = model(input_tensor)

    return albedo, shading, shadow, specular