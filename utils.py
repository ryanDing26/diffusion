import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import matplotlib.pyplot as plt
from dataloader import load_dataloaders
from diffusion import Diffusion
train_loader, test_loader = load_dataloaders()

# Step 1: get first batch
batch = next(iter(train_loader))
img = batch[0]  # shape C,H,W in [-1,1]

def show_image(img, output="test"):
    # img: C,H,W or H,W,C, float [-1,1] or [0,1] or uint8

    # 1. move to CPU
    img = img.detach().cpu()

    # 2. if in [-1,1], convert to [0,1]
    if img.min() < 0:
        img = (img + 1) / 2

    # 3. convert CHW → HWC
    if img.ndim == 3 and img.shape[0] in [1,3]:  
        img = img.permute(1, 2, 0)

    # 4. clamp to valid display range
    img = img.clamp(0, 1)

    # 5. convert tensor → numpy
    img_np = img.numpy()

    # 6. show
    plt.axis("off")
    plt.imsave(f"{output}.png", img_np)

def get_noisy_image(img, t):
    # add noise
    model = Diffusion()
    img_noisy, _ = model.q_sample(img, 50)
    img_noisier = model.q_sample(img, 200)

    # turn back into PIL image
    show_image(img, output="original")
    show_image(img_noisy, output="noisy")
    show_image(img_noisier, output="noisier")