import timm
import torch.nn as nn

class DecompoVisionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = timm.create_model('vit_base_patch16_224', prettrained=True)
        self.encoder.head = nn.Identity()

    def forward(self, x):
        features = self.encoder(x)
        return features