import torch

print("PyTorch version:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device("mps")

    x = torch.ones(3, 3, device=device)

    print("\nMPS test successful.")
    print("Device:", device)
    print(x)

else:
    print("\nMPS is not available.")