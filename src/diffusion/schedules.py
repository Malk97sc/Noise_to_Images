import torch

def beta_schedule(T, start = 1e-4, end = 0.02):
    """
    Generates T betas between start and end.
    """
    return torch.linspace(start, end, T, dtype=torch.float32)