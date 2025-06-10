from src.inference import run_inference
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    image_path = "indoor_room.jpg"
    img, outputs = run_inference(image_path)
    titles = ["Input", "Albedo", "Shading", "Shadow", "Specular"]
    plt.figure(figsize=(15, 3))

    for i, (title, out) in enumerate(zip(titles, [img] + outputs)):
        plt.subplot(1, 5, i + 1)
        plt.imshow(out)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
