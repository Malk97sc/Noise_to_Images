import torch

def predict(xt, T, eps, alphas_bar):
    """
    Reconstruct x0 from xt.

    x_t
    T: timestep
    eps: noise used at forward step
    alphas_bar: cumulative alphas
    """
    sqrt_ab = torch.sqrt(alphas_bar[T])
    sqrt_one_minus = torch.sqrt(1 - alphas_bar[T])
    return (xt - sqrt_one_minus * eps) / sqrt_ab

def p_sample(xt, T, eps, betas, alphas, alphas_bar):
    """
    One reverse sampling step: x_{t-1} from x_t
    """

    beta_t = betas[T]
    alpha_t = alphas[T]
    ab_t = alphas_bar[T]

    x0_pred = predict(xt, T, eps, alphas_bar) #denoised prediction
    coef1 = torch.sqrt(1.0 / alpha_t) * (xt - (beta_t / torch.sqrt(1 - ab_t)) * eps)

    if T > 0:
        noise = torch.randn_like(xt)
        sigma_t = torch.sqrt(beta_t)
        return coef1 + sigma_t * noise
    else:
        return coef1

def predict_x0_ddpm(xt, t, eps_theta, alphas_bar):
    """
    Reconstruct x0 from xt and the noise predicted by the NN

    xt: (B, C, H, W)
    t: (B,)
    eps_theta: (B, C, H, W)
    alphas_bar
    """
    if t.dim() == 0:
        ab_t = alphas_bar[t].view(1, 1, 1, 1) #only one t
    else:
        ab_t = alphas_bar[t].view(1, 1, 1, 1) #t (B,)

    sqrt_ab = torch.sqrt(ab_t)
    sqrt_one_minus = torch.sqrt(1.0 - ab_t)
    return (xt - sqrt_one_minus * eps_theta) / sqrt_ab

def p_sample_ddpm(xt, t, eps_theta, betas, alphas, alphas_bar):
    """
    A reverse step of DDPM: x_{t-1} from x_t.

    xt: (B, C, H, W)
    t: timestep
    eps_theta: (B, C, H, W) noise predicted by the U-Net
    """
    beta_t = betas[t]
    alpha_t = alphas[t]
    ab_t = alphas_bar[t]

    coef1 = 1.0 / torch.sqrt(alpha_t) * (xt - (beta_t / torch.sqrt(1.0 - ab_t)) * eps_theta)

    if t > 0:
        noise = torch.randn_like(xt)
        sigma_t = torch.sqrt(beta_t)
        return coef1 + sigma_t * noise
    else: #the first img (t = 0)
        return coef1
