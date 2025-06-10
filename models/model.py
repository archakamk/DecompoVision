import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder

# Inside model.py
class DecompoVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        features = self.encoder(x)
        last_feature_map = features[-1]
        return self.decoder(last_feature_map)
