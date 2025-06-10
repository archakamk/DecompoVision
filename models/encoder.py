import torch
import torch.nn as nn
import timm

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

    def forward(self, x):
        B = x.size(0)
        features = self.vit.forward_features(x)

        tokens = features[:, 1:, :]
        feature_map = tokens.transpose(1, 2).reshape(B, 768, 14, 14)

        return feature_map