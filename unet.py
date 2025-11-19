import torch
from torch import nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 3) if in_ch != out_ch else nn.Conv2d(in_ch, out_ch, 1)
        self.nonlinearity = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        
        # Add time embedding
        h += self.time_mlp(t_emb)[:, :, None, None] 

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        residual = self.residual_conv(x)
        assert residual.shape == h.shape
        return residual + h


class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj_out = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        q = self.q(h).reshape(B, C, H*W).permute(0, 2, 1)  # B, HW, C
        k = self.k(h).reshape(B, C, H*W)  # B, C, HW
        w = torch.bmm(q, k) * (C ** -0.5)  # B, HW, HW
        w = F.softmax(w, dim=-1)

        v = self.v(h).reshape(B, C, H*W).permute(0, 2, 1)  # B, HW, C
        h = torch.bmm(w, v).permute(0, 2, 1).reshape(B, C, H, W)
        h = self.proj_out(h)
        return x + h


class Downsample(nn.Module):
    def __init__(self, ch, with_conv=True):
        super().__init__()
        self.with_conv = with_conv

        if with_conv:
            self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
        else:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.conv(x) if self.with_conv else self.pool(x)



class Upsample(nn.Module):
    def __init__(self, ch, with_conv=True):
        super().__init__()
        self.with_conv = with_conv

        if with_conv:
            self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        # doubles height and width
        x = F.interpolate(x, size=(H * 2, W * 2), mode="nearest")
        return self.conv(x) if self.with_conv else x


class UNet(nn.Module):
    def __init__(self, 
                 in_ch=3,
                 out_ch=3,
                 base_ch=128, 
                 ch_mult=(1,1,2,2,4,4), 
                 num_res_blocks=2,
                 attn_resolutions=(16,),
                 dropout=0.0,
                ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base_ch = base_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.time_emb_dim = base_ch * 4

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )

        # Input conv
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_ch
        for i, mult in enumerate(ch_mult):
            out_ch_block = base_ch * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(ch, out_ch_block, self.time_emb_dim, dropout))
                ch = out_ch_block
            if i != len(ch_mult) - 1:
                self.downsamples.append(Downsample(ch))

        # Middle
        self.mid_block1 = ResidualBlock(ch, ch, self.time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch, self.time_emb_dim, dropout)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch_block = base_ch * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(ch*2, out_ch_block, self.time_emb_dim, dropout))
                ch = out_ch_block
            if i != 0:
                self.upsamples.append(Upsample(ch))

        # Output conv
        self.conv_out = nn.Conv2d(ch, out_ch, 3, padding=1)

    def _init_weights(self):
        """
        Initialize lasy layer weights to zero (Fix Up Initialization); Done within DDPM.

        Rationale: make training stable without normalization layers by having each residual block start at 0.
        """
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

        for module in self.modules():
            if isinstance(module, ResidualBlock):
                nn.init.zeros_(module.conv2.weight)
                nn.init.zeros_(module.conv2.bias)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)

        # Input conv
        h = self.conv_in(x)

        # Down path
        hs = []
        ds_idx = 0
        for i, block in enumerate(self.down_blocks):
            h = block(h, t_emb)
            if i % self.num_res_blocks == self.num_res_blocks - 1 and ds_idx < len(self.downsamples):
                hs.append(h)
                h = self.downsamples[ds_idx](h)
                ds_idx += 1
            else:
                hs.append(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Up path
        us_idx = 0
        for i, block in enumerate(self.up_blocks):
            res = hs.pop()
            h = torch.cat([h, res], dim=1)
            h = block(h, t_emb)
            if i % (self.num_res_blocks + 1) == self.num_res_blocks and us_idx < len(self.upsamples):
                h = self.upsamples[us_idx](h)
                us_idx += 1

        # Output
        return self.conv_out(h)
