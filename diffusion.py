import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from unet2 import UNet

class Diffusion:
    def __init__(self):
        # model params
        self.model = UNet(dim=256, channels=3, dim_mults=(1, 2, 4,))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # input sizes
        self.batch_size = 4
        self.channels = 3
        self.image_size = 256

        # diffusion constants
        self.timesteps = 10000
        self.betas = self.get_beta_schedule(timesteps=self.timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def get_beta_schedule(self, timesteps, schedule="cosine"):
        if schedule == "linear": # linear schedule
            start = 0.0001
            end = 0.02
            return torch.linspace(start, end, timesteps)
        else: # cosine schedule
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + steps) / (1 + steps) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)

    def _extract(self, vals, t, x_shape):
        """
        Return specific index t of a past list of values vals (batched for use in training)
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1, ) * (len(x_shape) - 1))).to(self.device)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Given image x_0 return noisy version x_t (batched)

        x_t = sqrt(alpha_t)*x_0 + sqrt(1 - alpha_t)*epsilon

        Returns mean and variance for gaussian sampling
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # sqrt(alpha_t)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)

        # sqrt(1 - alpha_t)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # (x_t, noise)
        return sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device) \
        + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device), noise.to(self.device)
    
    def p_losses(self, x_0, t, loss_type="l2"):
        """
        Given actual and predicted noise, compute loss between those
        """
        
        x_noisy, noise = self.q_sample(x_0, t)
        predicted_noise = self.model(x_noisy, t)

        if loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        
        return loss 
    
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self):
        # start from x_T aka pure noise
        img = torch.randn(self.batch_size, device=self.device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc="sampling loop time step", total=self.timesteps):
            img = self.p_sample(img, torch.full((self.batch_size,), i, device=self.device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())

        return imgs
                      
    @torch.no_grad()
    def sample(self):
        return self.p_sample_loop(shape=(self.batch_size, self.channels, self.image_size, self.image_size))