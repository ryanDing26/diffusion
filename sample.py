import torch
import imageio
import numpy as np
import torchvision.utils as vutils
from diffusion import Diffusion

model = Diffusion(timesteps=1000)
checkpoint = torch.load("./checkpoint_epoch0_1k.pt")
model.model.load_state_dict(checkpoint["model_state_dict"])

frames = model.sample()  # list of [B, C, H, W] numpy arrays

gif_frames = []
for t in range(len(frames)):
    img = frames[t][0]  # take sample index 0
    img_tensor = torch.tensor(img)

    # convert to uint8 [0,255]
    img_grid = vutils.make_grid(img_tensor, normalize=True)
    img_np = (img_grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    gif_frames.append(img_np)

imageio.mimsave("diffusion_trajectory.gif", gif_frames, fps=20)
