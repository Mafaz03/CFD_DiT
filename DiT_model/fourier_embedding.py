import torch


class MultiScaleFourierPositionEmbedding2D(torch.nn.Module):
    """
    Multi-scale random Fourier feature embedding for 2D coordinates in
    [0, 1] x [0, 1] (a 1m x 1m domain, matching your grid).

    Produces one embedding vector per patch center, to be added to patch
    tokens exactly where DiT's PatchEmbedding currently adds
    `get_patch_position_embedding(...)`.

    Args:
        d_model: transformer embedding dimension (must match DiT's d_model)
        num_frequencies: total number of random frequencies, split evenly
            across `sigmas` bands. Must be divisible by len(sigmas).
        sigmas: frequency scales (std of the Gaussian the frequency matrix
            is drawn from). Low sigma -> smooth/low-frequency content,
            high sigma -> sharp/high-frequency content (boundary layers,
            corner vortices). Multiple scales let one embedding cover both.
    """

    def __init__(self, d_model: int, num_frequencies: int = 128,
                 sigmas=(1.0, 2.0, 4.0)):
        super().__init__()
        assert num_frequencies % len(sigmas) == 0, \
            "num_frequencies must be divisible by len(sigmas)"

        self.n_bands = len(sigmas)
        self.freqs_per_band = num_frequencies // self.n_bands

        # B: (2, num_frequencies) -- one row per coordinate (x, y)
        B = torch.cat(
            [torch.randn(2, self.freqs_per_band) * s for s in sigmas], dim=1
        )

        self.progress = 1.0
        
        # Fixed, not trained -- only the projection below is learned.
        self.register_buffer("B", B)

        total_freq = B.shape[1]
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(2 * total_freq, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model),
        )

        torch.nn.init.normal_(self.proj[0].weight, std=0.02)
        torch.nn.init.constant_(self.proj[0].bias, 0)
        torch.nn.init.normal_(self.proj[2].weight, std=0.02)
        torch.nn.init.constant_(self.proj[2].bias, 0)

    def _band_mask(self, device):
        """
        (total_freq,) mask, one weight per frequency, low bands unlocking
        first as self.progress goes 0 -> 1. At progress=1.0 this is all
        ones, i.e. identical to the original non-annealed embedding.
        """
        mask = torch.zeros(self.n_bands * self.freqs_per_band, device=device)
        band_start = torch.linspace(0, 1, self.n_bands + 1, device=device)[:-1]
        ramp_steepness = 5.0
        for i in range(self.n_bands):
            lo, hi = i * self.freqs_per_band, (i + 1) * self.freqs_per_band
            alpha = torch.clamp((self.progress - band_start[i]) * ramp_steepness, 0.0, 1.0)
            mask[lo:hi] = alpha
        return mask
    
    def forward(self, grid_size, device):
        """
        grid_size: (num_patches_h, num_patches_w)
        Returns: (num_patches_h * num_patches_w, d_model)
        """
        gh, gw = grid_size

        # Patch *centers* in physical [0, 1] coordinates, not raw pixel
        # indices -- this is what makes the embedding calibrated to your
        # actual 1m x 1m domain regardless of grid resolution.
        ys = (torch.arange(gh, dtype=torch.float32, device=device) + 0.5) / gh
        xs = (torch.arange(gw, dtype=torch.float32, device=device) + 0.5) / gw
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (N, 2) flatennig and stacking

        proj = 2 * torch.pi * coords @ self.B  # (N, num_frequencies)
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, 2*num_frequencies)
        return self.proj(feats)  # (N, d_model)

