import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    image = tensor.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image * 255).astype(np.uint8)
    return image

def save_as_png(tensor, filename):
    image = tensor_to_image(tensor)
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))