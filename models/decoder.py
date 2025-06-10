import torch.nn as nn
import timm
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.upsample1 = nn.ConvTranspose2d(in_channels=768, out_channels=512, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.albedo_head = nn.Conv2d(64, 3, kernel_size=1)
        self.shading_head = nn.Conv2d(64, 3, kernel_size=1)
        self.shadow_head = nn.Conv2d(64, 3, kernel_size=1)
        self.specular_head = nn.Conv2d(64, 3, kernel_size=1)
    
    def forward(self, x):
        x = F.relu(self.upsample1(x))
        x = F.relu(self.upsample2(x))
        x = F.relu(self.upsample3(x))
        x = F.relu(self.upsample4(x))

        albedo = self.albedo_head(x)
        shading =self.shading_head(x)
        shadow = self.shadow_head(x)
        specular = self.specular_head(x)

        return albedo, shading, shadow, specular