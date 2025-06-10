'''import torch
import torchvision.transforms as T
import cv2
from models.model import DecompoVisionModel

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

def run_inference(image_path):
    model = DecompoVisionModel()
    model.eval()

    input_tensor = preprocess_image(image_path)

    with torch.no_grad():
        albedo, shading, shadow, specular = model(input_tensor)

    return albedo, shading, shadow, specular'''

# utils/inference.py

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
