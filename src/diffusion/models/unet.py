import torch
import torch.nn as nn
from diffusion.modules.time_embedding import TimeEmbedding

def conv3x3(in_c, out_c):
    return nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

def choose_groups(channels, default=8):
    groups = min(default, channels)
    while groups > 1 and (channels % groups != 0):
        groups -= 1
    return max(1, groups)

class ResBlock(nn.Module):
    """
    Residual block that conditions on a time embedding
    """
    def __init__(self, in_c, out_c, time_emb_dim=None, gn_groups=8):
        super().__init__()
        
        self.conv1 = conv3x3(in_c, out_c)
        groups = choose_groups(out_c, gn_groups) #it's only for test
        self.norm1 = nn.GroupNorm(groups, out_c) #the same as GroupNorm(min(8, out_c)) 
        self.act = nn.SiLU()

        self.conv2 = conv3x3(out_c, out_c)
        self.norm2 = nn.GroupNorm(groups, out_c)

        self.use_time = time_emb_dim is not None
        if self.use_time:
            self.time_proj = nn.Linear(time_emb_dim, out_c)

        if in_c != out_c:
            self.skip = nn.Conv2d(in_c, out_c, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward (self, x, t_emb = None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        if self.use_time:
            t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1) #to be (B, out_c, 1, 1)
            h = h + t
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.skip(x)
    

class Down(nn.Module):
    """Downscaling with maxpool and time embedding"""
    def __init__(self, in_c, out_c, time_emb_dim=None):
        super().__init__()
        self.block = ResBlock(in_c, out_c, time_emb_dim=time_emb_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t_emb):
        h = self.block(x, t_emb)
        return self.pool(h), h
    
class Up(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim=None):
        super().__init__()
        #in_c needs to concatenated channels. That's for the skip connection
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.block = ResBlock(in_c, out_c, time_emb_dim=time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        #if spatial sizes mismatch by 1 due to odd sizes, center-crop skip
        if x.shape[-2:] != skip.shape[-2:]:
            sh, sw = skip.shape[-2], skip.shape[-1]#crop skip to x size
            th, tw = x.shape[-2], x.shape[-1]
            dh = (sh - th) // 2
            dw = (sw - tw) // 2
            skip = skip[:, :, dh:dh+th, dw:dw+tw]
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t_emb)
    
class UNet(nn.Module):
    """
    U-Net for diffusion with time embedding conditioning.
    img_channels: input/output channels
    base_c: base channels
    time_dim: dimension from TimeEmbedding
    """
    def __init__(self, img_channels = 3, base_c = 32, time_dim = 128):
        super().__init__()

        self.time_embed = TimeEmbedding(dim=32, hidden_dim=time_dim) #sinusoidal dim 32 for test

        #encoder block
        self.down1 = Down(img_channels, base_c, time_emb_dim = time_dim)
        self.down2 = Down(base_c, base_c * 2, time_emb_dim = time_dim)
        self.down3 = Down(base_c * 2, base_c * 4, time_emb_dim = time_dim)
        self.down4 = Down(base_c * 4, base_c * 8, time_emb_dim = time_dim)

        #bottleneck
        self.bott = ResBlock(base_c * 8, base_c * 8, time_emb_dim = time_dim)

        #decoder
        self.up1 = Up(base_c * 8 + base_c * 8, base_c * 4, time_emb_dim = time_dim)
        self.up2 = Up(base_c * 4 + base_c * 4, base_c * 2, time_emb_dim = time_dim)
        self.up3 = Up(base_c * 2 + base_c * 2, base_c, time_emb_dim = time_dim)
        self.up4 = Up(base_c + base_c , base_c, time_emb_dim = time_dim)

        self.out = nn.Conv2d(base_c, img_channels, kernel_size=1)

    def forward(self, x, t):
        """
        x: (B, C, H, W)
        t: (B,) timesteps
        returns: (B, channels, H, W)
        """
        t_emb = self.time_embed(t)

        d1, s1 = self.down1(x, t_emb)
        d2, s2 = self.down2(d1, t_emb)
        d3, s3 = self.down3(d2, t_emb)
        d4, s4 = self.down4(d3, t_emb)

        bott = self.bott(d4, t_emb)

        u1 = self.up1(bott, s4, t_emb)
        u2 = self.up2(u1, s3, t_emb)
        u3 = self.up3(u2, s2, t_emb)
        u4 = self.up4(u3, s1, t_emb)

        return self.out(u4)
    
""" Minimal Test
if __name__ == "__main__":
    device = torch.device("cpu")
    model = UNet(img_channels=1, base_c=32, time_dim=128).to(device)
    x = torch.randn(2, 1, 64, 64).to(device) # batch=2, 64x64 images
    t = torch.randint(0, 1000, (2,), dtype=torch.long).to(device) #timesteps
    y = model(x, t)
    print("input:", x.shape, "output:", y.shape)
    #expected output: (2, 1, 64, 64)
"""