import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(768, 512, kernel_size=5, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=5, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=5, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=5, padding=1),
            nn.ReLU(inplace=True)
        )

        self.albedo_head = nn.Conv2d(64, 3, kernel_size=1)
        self.shading_head = nn.Conv2d(64, 3, kernel_size=1)
        self.shadow_head = nn.Conv2d(64, 3, kernel_size=1)
        self.specular_head = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        albedo = self.albedo_head(x)
        shading = self.shading_head(x)
        shadow = self.shadow_head(x)
        specular = self.specular_head(x)

        return albedo, shading, shadow, specular