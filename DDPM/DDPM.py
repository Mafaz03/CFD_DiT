import torch
from tqdm import tqdm
from torchvision.utils import make_grid
import torchvision

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)

        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)

    def get_x0(self, xt, pred, t, clamp_range=None):
        sqrt_one_minus_ac = self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t].view(-1, 1, 1, 1)
        sqrt_ac = torch.sqrt(self.alpha_cum_prod.to(xt.device)[t]).view(-1, 1, 1, 1)
        x0 = (xt - sqrt_one_minus_ac * pred) / sqrt_ac
        if clamp_range is not None:
            x0 = torch.clamp(x0, *clamp_range)
        return x0

    def sample_prev_timestep(self, xt, pred, t):
        x0 = self.get_x0(xt, pred, t)

        mean = xt - ((self.betas.to(xt.device)[t]) * pred) / (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])

        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0

    def ddim_step(self, xt, pred_noise, t, t_prev, eta=0.0):
        alpha_t = self.alpha_cum_prod.to(xt.device)[t]
        alpha_prev = self.alpha_cum_prod.to(xt.device)[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=xt.device)

        x0_pred = (xt - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        x0_pred = torch.clamp(x0_pred, -4, 4)  # match your data's actual range, not [-1,1] — see note below

        dir_xt = torch.sqrt(1 - alpha_prev) * pred_noise
        xt_prev = torch.sqrt(alpha_prev) * x0_pred + dir_xt
        return xt_prev, x0_pred