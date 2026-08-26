import torch
from diffusers.models import AutoencoderKL

### will be using pretrained VAE
### but code here incase we re train VAE

SD_VAE_SCALING_FACTOR = 0.18215  

class VAE(torch.nn.Module):
    def __init__(self, device, freeze: bool, scaling_factor = None, path = "stabilityai/sd-vae-ft-mse", load_weights = None):
        super().__init__()

        self.vae: torch.nn.Module = AutoencoderKL.from_pretrained(path)
        
        if load_weights: 
            self.vae.load_state_dict(torch.load(load_weights, map_location = device))
            print("[VAE] weights loaded!")

        self.vae = self.vae.to(device)

        self.scaling_factor = SD_VAE_SCALING_FACTOR if not scaling_factor else scaling_factor

        if freeze:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

    def encode(self, normal_grid):
        posterior = self.vae.encode(normal_grid).latent_dist
        mu     = posterior.mean   * self.scaling_factor
        logvar = posterior.logvar + 2 * torch.log(torch.tensor(self.scaling_factor, device=normal_grid.device))
        return mu, logvar

    def decode(self, compressed_grid):
        z = compressed_grid / self.scaling_factor
        reconstructed_grid = self.vae.decode(z)
        return reconstructed_grid.sample

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, grid):
        mu, logvar = self.encode(normal_grid = grid)        # each [B, 4, H/8, W/8]
        z = self.reparameterize(mu = mu, logvar = logvar)   # sampling from gausian distribution
        recon_grid = self.decode(compressed_grid = z)
        return recon_grid, mu, logvar


if __name__ == "__main__":
    img = torch.rand(3, 3, 256, 256)
    vae = VAE(device = "cpu", freeze = True, scaling_factor = SD_VAE_SCALING_FACTOR)
    grid, mu, logvar = vae(img)
    print(grid.shape)
    print(mu.shape)
    print(logvar.shape)
