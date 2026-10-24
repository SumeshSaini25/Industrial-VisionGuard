import torch
from anomalib.models import Patchcore

print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
)

print("\nPatchCore model created successfully.")
print("Backbone: wide_resnet50_2")
print("Layers: layer2, layer3")
print("Coreset sampling ratio: 0.1")