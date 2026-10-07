import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import shift

import numpy as np

from scipy.interpolate import griddata

import json

ROOT = Path(__file__).resolve().parent.parent

with open(f"{ROOT}/config/config.json", "r") as file:
    config = json.load(file)

import numpy as np
from scipy.ndimage import binary_erosion


def to_unit_range(x, lo, hi, **kwargs):   return 2.0 * (x - lo) / (hi - lo) - 1.0
def from_unit_range(y, lo, hi, **kwargs): return (y + 1.0) / 2.0 * (hi - lo) + lo


def get_boundary_coordinates(mask, u_grid, v_grid, U, V, tol=1e-1):
    """
    Returns boundary-condition masks with the same shape as the input grid.

    wall_mask:
        1 where boundary AND u = 0, v = 0
        0 elsewhere

    uv_mask:
        1 where boundary AND u = U, v = V
        0 elsewhere
    """

    # Find cells directly on the edge of the mask
    eroded = binary_erosion(mask.astype(bool))
    boundary = mask.astype(bool) & ~eroded

    # u = 0, v = 0
    wall = (
        np.isclose(u_grid, 0.0, atol=tol) &
        np.isclose(v_grid, 0.0, atol=tol)
    )

    # u = U, v = V
    uv = (
        np.isclose(u_grid, U, atol=tol) &
        np.isclose(v_grid, V, atol=tol)
    )

    # Only accept velocity conditions on the boundary
    wall_mask = (boundary & wall).astype(np.float32)
    uv_mask = (boundary & uv).astype(np.float32)

    return wall_mask, uv_mask

class dataset_csv(Dataset):

    def __init__(
        self,
        folder: str,
        meta: str,
        grid_size: int = 256,
        eps = 0.001
    ):
        self.folder = Path(folder)

        allowed_exts = {".csv"}

        self.all_pths = [
            self.folder / name
            for name in os.listdir(self.folder)
            if Path(name).suffix.lower() in allowed_exts
        ]

        self.grid_size = grid_size
        self.meta = meta
        self.eps = eps

        with open(f"{ROOT}/Data/Problems/{meta}.json", "r") as file:
            self.re_json = json.load(file)

    def __len__(self):
        return len(self.all_pths)

    def __getitem__(self, index):
        selected = self.all_pths[index]
        df = pd.read_csv(selected)

        n = self.grid_size
        cfg = config["Stats"][self.meta]

        # CSVs are already on the 256x256 canvas, so read them directly
        u_grid = df["u (m/s)"].values.astype(np.float32).reshape(n, n)
        v_grid = df["v (m/s)"].values.astype(np.float32).reshape(n, n)
        P_grid = df["p (Pa)"].values.astype(np.float32).reshape(n, n)

        # mask = flagged valid AND all fields finite
        mask = df["mask"].values.astype(bool).reshape(n, n)
        mask &= ~np.isnan(u_grid) & ~np.isnan(v_grid) & ~np.isnan(P_grid)
        mask = mask.astype(np.float32)
        inside = mask > 0.5

        # Clip (NaN stays NaN)
        u_grid = np.clip(u_grid, cfg["U_CLIP_MIN"], cfg["U_CLIP_MAX"])
        v_grid = np.clip(v_grid, cfg["V_CLIP_MIN"], cfg["V_CLIP_MAX"])
        P_grid = np.clip(P_grid, cfg["P_CLIP_MIN"], cfg["P_CLIP_MAX"])

        # Boundary masks (isclose(NaN, 0) is False, so outside never counts as wall)
        wall_xy, uv_xy = get_boundary_coordinates(mask, u_grid, v_grid, U=1.0, V=0.0)

        # Normalize, then fill outside with a neutral value
        u_grid = to_unit_range(u_grid, cfg["U_CLIP_MIN"], cfg["U_CLIP_MAX"])
        v_grid = to_unit_range(v_grid, cfg["V_CLIP_MIN"], cfg["V_CLIP_MAX"])
        P_grid = to_unit_range(P_grid, cfg["P_CLIP_MIN"], cfg["P_CLIP_MAX"])

        u_grid = np.where(inside, u_grid, 0.0)
        v_grid = np.where(inside, v_grid, 0.0)
        P_grid = np.where(inside, P_grid, 0.0)

        uvp_grid = np.stack([u_grid, v_grid, P_grid], axis=0).astype(np.float32)

        # --------------------------------------------------
        # Random shift inside the empty margins
        # --------------------------------------------------
        ys, xs = np.where(mask > 0.5)

        if len(ys) > 0:
            # Free space (in pixels) between the valid region and each canvas edge
            space = {
                "left":  xs.min(),                 # towards x = 0
                "right": n - 1 - xs.max(),         # towards x = 1
                "down":  ys.min(),                 # towards y = 0 (row 0)
                "up":    n - 1 - ys.max(),         # towards y = 1 (last row)
            }

            # Only directions that actually have room
            options = [d for d, s in space.items() if s > 0]

            if options:
                direction = np.random.choice(options)
                amount = np.random.randint(1, space[direction] + 1)

                dx, dy = 0, 0
                if direction == "left":
                    dx = -amount
                elif direction == "right":
                    dx = amount
                elif direction == "down":
                    dy = -amount
                else:  # up
                    dy = amount

                # Same shift for every array so they stay aligned
                uvp_grid = np.roll(uvp_grid, shift=(dy, dx), axis=(1, 2))
                mask     = np.roll(mask,     shift=(dy, dx), axis=(0, 1))
                wall_xy  = np.roll(wall_xy,  shift=(dy, dx), axis=(0, 1))
                uv_xy    = np.roll(uv_xy,    shift=(dy, dx), axis=(0, 1))

        number = self.re_json[selected.stem]

        return (
            uvp_grid,
            (number - cfg["Re_Mean"]) / cfg["Re_Std"],
            mask,
            wall_xy,
            uv_xy
        )

if __name__ == "__main__":

    dataset = dataset_csv(
        folder="Data/Problems/Lid_Driven_domain",
        meta="Lid_Driven"
    )

    # dataset = dataset_csv(
    #         folder="Data/Problems/re_full_domain",
    #         meta="re_full"
    #     )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True
    )

    uvp_grid, number, mask, wall_mask, uv_mask = next(iter(dataloader))

    u = uvp_grid[0, 0].numpy()
    v = uvp_grid[0, 1].numpy()
    P = uvp_grid[0, 2].numpy()

    u = from_unit_range(
        uvp_grid[0, 0].numpy(),
        config["Stats"]["Lid_Driven"]["U_CLIP_MIN"],
        config["Stats"]["Lid_Driven"]["U_CLIP_MAX"]
    )

    v = from_unit_range(
        uvp_grid[0, 1].numpy(),
        config["Stats"]["Lid_Driven"]["V_CLIP_MIN"],
        config["Stats"]["Lid_Driven"]["V_CLIP_MAX"]
    )

    P = from_unit_range(
        uvp_grid[0, 2].numpy(),
        config["Stats"]["Lid_Driven"]["P_CLIP_MIN"],
        config["Stats"]["Lid_Driven"]["P_CLIP_MAX"]
    )
    
    print((number * config["Stats"]["Lid_Driven"]["Re_Std"]) + config["Stats"]["Lid_Driven"]["Re_Mean"], P.min(), P.max())

    m = mask[0].numpy()

    mag = np.sqrt(u**2 + v**2)

    x_grid = np.linspace(0, 1, u.shape[1])
    y_grid = np.linspace(0, 1, u.shape[0])
    X, Y = np.meshgrid(x_grid, y_grid)

    u_plot = u.copy()
    v_plot = v.copy()
    P_plot = P.copy()
    mag_plot = mag.copy()

    u_plot[m < 0.5] = np.nan
    v_plot[m < 0.5] = np.nan
    P_plot[m < 0.5] = np.nan
    mag_plot[m < 0.5] = np.nan

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    cf = axes[0, 0].contourf(X, Y, u_plot, levels=50, cmap="jet")
    axes[0, 0].set_title("U Velocity")
    axes[0, 0].set_aspect("equal")
    fig.colorbar(cf, ax=axes[0, 0])

    cf = axes[0, 1].contourf(X, Y, v_plot, levels=50, cmap="jet")
    axes[0, 1].set_title("V Velocity")
    axes[0, 1].set_aspect("equal")
    fig.colorbar(cf, ax=axes[0, 1])

    cf = axes[0, 2].contourf(X, Y, P_plot, levels=50, cmap="jet")
    axes[0, 2].set_title("Pressure")
    axes[0, 2].set_aspect("equal")
    fig.colorbar(cf, ax=axes[0, 2])

    cf = axes[1, 0].contourf(X, Y, mag_plot, levels=50, cmap="jet")
    axes[1, 0].set_title("Velocity Magnitude")
    axes[1, 0].set_aspect("equal")
    fig.colorbar(cf, ax=axes[1, 0])

    cf = axes[1, 1].contourf(X, Y, m, levels=[0, 0.5, 1], cmap="gray")
    axes[1, 1].set_title("CFD Mask")
    axes[1, 1].set_aspect("equal")
    fig.colorbar(cf, ax=axes[1, 1])

    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()