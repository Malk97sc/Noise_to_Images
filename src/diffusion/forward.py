import torch

def compute_alphas(betas):
    """
    Calculate all the betas operations:
    - alphas: 1-betas
    - alphas_bar: cumulative product of alphas
    """

    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return alphas, alphas_bar

def q_sample(x0, t, alphas_bar):
    """
    Calculate x_t
    
    x0: image tensor
    t: timestep (0 <= t < T)
    alphas_bar

    returns x_t
    """
    noise = torch.randn_like(x0)
    sqrt_ab = torch.sqrt(alphas_bar[t])
    sqrt_one_minus_ab = torch.sqrt(1 - alphas_bar[t])
    return sqrt_ab * x0 + sqrt_one_minus_ab * noise

def q_sample_ddpm(x0, t, alphas_bar):
    """
    Batch Function to Forward diffusion q(x_t | x_0)

    x0: (B, C, H, W)
    t: (B,) timesteps
    alphas_bar

    returns:
        x_t: (B, C, H, W)
        noise: (B, C, H, W) to build x_t
    """
    noise = torch.randn_like(x0)
    ab_t = alphas_bar[t].view(-1, 1, 1, 1)  #(B,1,1,1) 
    x_t = torch.sqrt(ab_t) * x0 + torch.sqrt(1.0 - ab_t) * noise
    return x_t, noise