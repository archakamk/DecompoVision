import os
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

class IntrinsicImageDataset(Dataset):
    def __init__(self, root, dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.input_images = sorted(os.listdir(os.path.join(root_dir, "input")))

    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):
        pass
