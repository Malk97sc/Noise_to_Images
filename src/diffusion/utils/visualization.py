import torch

def tensor2image(x):
    """
    Convert a tensor (x) in a RGB image
    """
    x = x.detach().cpu().clamp(0, 1)
    return x.permute(1, 2, 0).numpy()