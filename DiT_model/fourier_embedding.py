"""
Fourier feature components for adapting the DiT model (see DiT.py) into a
Fourier-feature-enhanced surrogate for lid-driven cavity flow.

Why this exists
----------------
Standard MLP/transformer patch-position embeddings using a *fixed* sinusoidal
schedule (as in `get_patch_position_embedding` in DiT.py) still suffer from
"spectral bias" -- the network preferentially fits low-frequency spatial
structure and struggles with sharp gradients (e.g. the corner vortices and
thin boundary layers near the walls in a lid-driven cavity).

`MultiScaleFourierPositionEmbedding2D` replaces that fixed embedding with a
random Fourier feature embedding at several frequency scales (Tancik et al.,
2020 "Fourier Features Let Networks Learn High Frequency Functions"; Wang et
al., 2021 "On the eigenvector bias of Fourier feature networks" -- the same
idea behind "Fourier PINNs"). The frequency matrix B is fixed (non-trainable)
per band; a small learnable MLP projects the resulting sin/cos features to
d_model, so the network can still learn how much to weight each frequency
band.

This is NOT a PDE-residual loss. Your setup is supervised (ground truth CFD
data), so the only "physics" injected here is the exact Dirichlet boundary
conditions of the lid-driven cavity, applied as an explicit loss term.
"""

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
                 sigmas=(1.0, 10.0, 50.0)):
        super().__init__()
        assert num_frequencies % len(sigmas) == 0, \
            "num_frequencies must be divisible by len(sigmas)"

        freqs_per_band = num_frequencies // len(sigmas)
        # B: (2, num_frequencies) -- one row per coordinate (x, y)
        B = torch.cat(
            [torch.randn(2, freqs_per_band) * s for s in sigmas], dim=1
        )
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


def lid_cavity_bc_loss(field: torch.Tensor, u_lid: float = 1.0,
                        lid_edge: str = "top", channels=("u", "v")):
    """
    Explicit Dirichlet boundary-condition penalty for a lid-driven cavity.

    field: (B, C, H, W) predicted flow field. Assumes channel order
        (u, v, p) by default -- pass `channels` to reorder/rename if yours
        differs, only "u" and "v" positions are used here.
    u_lid: horizontal velocity of the moving lid (e.g. 1.0 in
        non-dimensional units).
    lid_edge: which edge of the HxW grid is the moving lid --
        "top" (row 0), "bottom" (row -1), "left" (col 0), "right" (col -1).
        Verify this matches your data's (row, col) <-> (y, x) convention.
    channels: names of the channel dimension, e.g. ("u", "v", "p").
        Only used to find the indices of "u" and "v".

    Returns: scalar MSE loss over all four boundary edges.
    """
    u_idx = channels.index("u")
    v_idx = channels.index("v")
    u = field[:, u_idx]  # (B, H, W)
    v = field[:, v_idx]  # (B, H, W)

    loss = 0.0
    n_terms = 0

    edges = {
        "top":    (u[:, 0, :],  v[:, 0, :]), 
        "bottom": (u[:, -1, :], v[:, -1, :]),
        "left":   (u[:, :, 0],  v[:, :, 0]),
        "right":  (u[:, :, -1], v[:, :, -1]),
    }

    for edge_name, (u_edge, v_edge) in edges.items():
        target_u = u_lid if edge_name == lid_edge else 0.0
        loss = loss + torch.mean((u_edge - target_u) ** 2)
        loss = loss + torch.mean(v_edge ** 2)  # v = 0 on all solid/lid edges
        n_terms += 2

    return loss / n_terms
