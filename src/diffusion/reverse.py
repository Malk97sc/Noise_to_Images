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