import torch
from torch.optim import Adam
from unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet()
model.to(device)
optimizer = Adam(model.parameters(), lr=2e-5)