import torch
from encoder import Encoder

encoder = Encoder()
encoder.eval()  # Set to eval mode since pretrained weights

dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, RGB image 224x224

features = encoder(dummy_input)

print(f"Number of feature maps: {len(features)}")
for i, f in enumerate(features):
    print(f"Feature map {i} shape: {f.shape}")
