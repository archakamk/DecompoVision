import torch
from utils.preprocess import preprocess_image_cv2
from models.model import DecompoVisionModel
import matplotlib.pyplot as plt

def run_inference(image_path):
    model = DecompoVisionModel()
    model.eval()

    input_tensor = preprocess_image_cv2(image_path)

    with torch.no_grad():
        albedo, shading, shadow, specular = model(input_tensor)

    return input_tensor, albedo, shading, shadow, specular

def show_tensor(tensor, title):
    img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)  # Normalize for display
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

def visualize_image(input_tensor, albedo, shading, shadow, specular):
    plt.figure(figsize=(15, 8))

    # Input image: convert tensor to numpy image for display
    input_img = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    input_img = (input_img * 0.5 + 0.5).clip(0,1)  # assuming normalization was mean=0.5, std=0.5
    plt.subplot(2, 3, 1)
    plt.imshow(input_img)
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

if __name__ == "__main__":
    img_path = "data/indoor_room.jpg"
    input_tensor, albedo, shading, shadow, specular = run_inference(img_path)

    # Save outputs if needed
    '''save_as_png(albedo, "albedo.png")
    save_as_png(shading, "shading.png")
    save_as_png(shadow, "shadow.png")
    save_as_png(specular, "specular.png")'''

    visualize_image(input_tensor, albedo, shading, shadow, specular)