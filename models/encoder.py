import torch.nn as nn
import timm

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True, features_only=True)
    
    def forward(self, x):
        return self.encoder(x)