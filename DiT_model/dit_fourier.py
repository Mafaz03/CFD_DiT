"""
Fourier-feature variant of your DiT model for the lid-driven cavity
regression task.

Reuses everything from DiT.py (TransformerLayer, get_time_embedding,
NumberEmbedding) and only swaps the *patch position embedding* for the
multi-scale random Fourier feature embedding in fourier_embedding.py.
Nothing in DiT.py is modified.

Note: DiT.py reads `{ROOT}/config/config.json` at import time (ROOT = two
dirs above DiT.py). Make sure that file exists relative to wherever this
script lives, or importing DiT will raise FileNotFoundError.
"""

import torch
from einops import rearrange

from DiT import TransformerLayer, get_time_embedding, NumberEmbedding
from fourier_embedding import MultiScaleFourierPositionEmbedding2D


class FourierPatchEmbedding(torch.nn.Module):
    """
    Same as DiT.PatchEmbedding, except the additive position embedding is
    the learned multi-scale Fourier feature embedding instead of the fixed
    sinusoidal grid embedding.
    """

    def __init__(self, grid_height, grid_width, g_channels, patch_height,
                 patch_width, d_model, num_frequencies=128,
                 sigmas=(1.0, 10.0, 50.0)):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.g_channels = g_channels
        self.d_model = d_model
        self.patch_height = patch_height
        self.patch_width = patch_width

        patch_dim = self.g_channels * self.patch_height * self.patch_width
        self.patch_embed = torch.nn.Sequential(torch.nn.Linear(patch_dim, self.d_model))

        torch.nn.init.xavier_uniform_(self.patch_embed[0].weight)
        torch.nn.init.constant_(self.patch_embed[0].bias, 0)

        self.fourier_pos_embed = MultiScaleFourierPositionEmbedding2D(
            d_model=d_model, num_frequencies=num_frequencies, sigmas=sigmas
        ) # output shape: num_patches_h * num_patches_w, d_model

    def forward(self, x):
        grid_size_h = self.grid_height // self.patch_height
        grid_size_w = self.grid_width // self.patch_width

        out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',
                         ph=self.patch_height, pw=self.patch_width)
        out = self.patch_embed(out)

        pos_embed = self.fourier_pos_embed(
            grid_size=(grid_size_h, grid_size_w), device=x.device
        )
        out = out + pos_embed
        return out


class FourierDiT(torch.nn.Module):
    """
    DiT with Fourier-feature patch positions, used as a direct conditional
    regressor: field = FourierDiT(x, t, n)

    x: (B, g_channels, H, W) -- for this supervised (non-diffusion) setup,
       feed a fixed placeholder (e.g. zeros or the domain mask), since the
       spatial signal comes entirely from the Fourier position embedding,
       not from x itself.
    t: fixed constant timestep tensor (e.g. torch.zeros(B)) -- kept only
       because TransformerLayer's conditioning MLP expects a t embedding;
       it carries no diffusion meaning here.
    n: Reynolds number (or other conditioning scalar), shape (B,)
    """

    def __init__(self, d_model, patch_size, grid_size, g_channels,
                 out_channels, timestep_emb_dim, number_emb_dim,
                 num_layers, num_heads, num_frequencies=128,
                 sigmas=(1.0, 10.0, 50.0)):
        super().__init__()

        self.grid_height = grid_size
        self.grid_width = grid_size
        self.g_channels = g_channels
        self.out_channels = out_channels
        self.d_model = d_model
        self.patch_height = patch_size
        self.patch_width = patch_size
        self.timestep_emb_dim = timestep_emb_dim
        self.number_emb_dim = number_emb_dim

        self.nh = self.grid_height // self.patch_height
        self.nw = self.grid_width // self.patch_width

        self.patch_embed_layer = FourierPatchEmbedding(
            grid_height     = self.grid_height, 
            grid_width      = self.grid_width,
            g_channels      = self.g_channels, 
            patch_height    = self.patch_height,
            patch_width     = self.patch_width, 
            d_model         = self.d_model,
            num_frequencies = num_frequencies, 
            sigmas          = sigmas,
        )

        self.t_proj = torch.nn.Sequential(
            torch.nn.Linear(self.timestep_emb_dim, self.d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(self.d_model, self.d_model),
        )
        self.number_embed = NumberEmbedding(self.number_emb_dim, self.d_model)

        self.layers = torch.nn.ModuleList([
            TransformerLayer(d_model=d_model, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        self.norm = torch.nn.LayerNorm(self.d_model, elementwise_affine=False, eps=1e-6)
        self.adaptive_norm_layer = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(self.d_model, 2 * self.d_model, bias=True),
        )

        # Output channels are independent of input channels (unlike the
        # original DiT, where in/out both equal g_channels). This lets x be
        # a placeholder / mask while the model still outputs (u, v, p).
        self.proj_out = torch.nn.Linear(
            self.d_model, self.patch_height * self.patch_width * self.out_channels
        )

        torch.nn.init.normal_(self.t_proj[0].weight, std=0.02)
        torch.nn.init.normal_(self.t_proj[2].weight, std=0.02)
        torch.nn.init.normal_(self.number_embed.proj[0].weight, std=0.02)
        torch.nn.init.normal_(self.number_embed.proj[2].weight, std=0.02)
        torch.nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        torch.nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)
        torch.nn.init.constant_(self.proj_out.weight, 0)
        torch.nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x, t, n):
        out = self.patch_embed_layer(x)

        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.timestep_emb_dim)
        n_emb = self.number_embed(n.float())
        c_emb = self.t_proj(t_emb) + n_emb

        for layer in self.layers:
            out = layer(out, c_emb)

        pre_mlp_shift, pre_mlp_scale = self.adaptive_norm_layer(c_emb).chunk(2, dim=1)
        out = (self.norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) + pre_mlp_shift.unsqueeze(1))

        out = self.proj_out(out)
        out = rearrange(out, 'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',
                         ph=self.patch_height, pw=self.patch_width,
                         nw=self.nw, nh=self.nh)
        return out


if __name__ == "__main__":
    model = FourierDiT(
        d_model=512, patch_size=2, grid_size=32,
        g_channels=1,          # placeholder input channel (e.g. domain mask)
        out_channels=3,        # (u, v, p)
        timestep_emb_dim=512, number_emb_dim=512,
        num_layers=12, num_heads=16,
        num_frequencies=129, sigmas=(1.0, 10.0, 50.0),
    )
    print(f"{sum(p.numel() for p in model.parameters()):,} params")

    B = 4
    x = torch.zeros(B, 1, 32, 32)            # placeholder input
    t = torch.zeros(B)                        # unused, fixed
    n = torch.tensor([100., 400., 1000., 3200.])  # Reynolds numbers
    out = model(x, t, n)
    print(out.shape)  # -> (4, 3, 32, 32)
