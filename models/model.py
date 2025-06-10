import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder

# Inside model.py
class DecompoVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder_albedo = Decoder()
        self.decoder_shading = Decoder()
        self.decoder_shadow = Decoder()
        self.decoder_specular = Decoder()

    def forward(self, x):
        features = self.encoder(x)
        albedo = self.decoder_albedo(features)
        shading = self.decoder_shading(features)
        shadow = self.decoder_shadow(features)
        specular = self.decoder_specular(features)
        return albedo, shading, shadow, specular
