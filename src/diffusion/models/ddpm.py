import torch
import torch.nn as nn

from diffusion.schedules import beta_schedule
from diffusion.forward import compute_alphas, q_sample_ddpm
from diffusion.reverse import p_sample_ddpm
from diffusion.models.unet import UNet

class DDPM(nn.Module):
    def __init__(self, img_channels=1, base_c=32, time_dim=128, T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = T

        #U-Net model
        self.model = UNet(img_channels=img_channels, base_c=base_c, time_dim=time_dim)
        
        #Schedules
        betas = beta_schedule(T, start=beta_start, end=beta_end)
        alphas, alphas_bar = compute_alphas(betas)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)

        self.mse = nn.MSELoss()

    def forward(self, x0):
        """
        x0: (B, C, H, W) in [-1, 1]
        """
        b = x0.size(0)
        device = x0.device

        #t for each element in the batch
        t = torch.randint(0, self.T, (b,), device=device).long()

        # q(x_t | x_0)
        x_t, noise = q_sample_ddpm(x0, t, self.alphas_bar)
        
        #noise prediction
        eps_theta = self.model(x_t, t)

        loss = self.mse(eps_theta, noise)
        return loss
    
    @torch.no_grad()
    def sample(self, n_samples, img_size=(1, 32, 32)):
        """
        Sample images from noise using the reverse process

        n_samples: n images
        img_size: (C, H, W)
        """
        self.model.eval()
        C, H, W = img_size

        device = next(self.model.parameters()).device
        x_t = torch.randn(n_samples, C, H, W, device=device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)
            eps_theta = self.model(x_t, t_batch)  #noise prediction

            x_t = p_sample_ddpm(x_t, t, eps_theta, self.betas, self.alphas, self.alphas_bar)

        x_0 = torch.clamp(x_t, -1.0, 1.0)
        return x_0
