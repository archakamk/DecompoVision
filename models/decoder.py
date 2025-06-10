import torch.nn as nn
import timm

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=2, stride=2),  # 14 -> 28
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 28 -> 56
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 56 -> 112
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 112 -> 224
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=1)
        )

    def forward(self, x):
        return self.decoder(x)