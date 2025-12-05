import torch 
import torch .nn as nn
import math

def sinusoidal_emb(timesteps, dim):
    """
    timesteps: (batch,) tensor
    returns: (batch, dim)
    """
    if timesteps.dim() == 2:
        timesteps = timesteps.squeeze(1)
    assert timesteps.dim() == 1, "timesteps should be shape (batch,)"

    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / float(half)
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb 

class TimeEmbedding(nn.Module):
    """
    Sinusoidal positional based in the timestep embedding
    - dim: dimension of the sinusoidal embedding
    - hidden_dim: output dimension (used by the network)
    """
    def __init__(self, dim= 32, hidden_dim = 128):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU() #SiLU works better in diffusion models
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, t):
        """
        t: (B,) tensor of timesteps
        returns: (B, hidden_dim)
        """
        emb = sinusoidal_emb(t, self.linear1.in_features)
        emb = self.linear1(emb)
        emb = self.act(emb)
        emb = self.linear2(emb)
        return emb
