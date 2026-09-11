import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

from scipy.interpolate import griddata

import json

ROOT = Path(__file__).resolve().parent.parent

with open(f"{ROOT}/config/config.json", "r") as file:
    config = json.load(file)

import numpy as np
from scipy.ndimage import binary_erosion


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
        grid_size: int = 256
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

    def __len__(self):
        return len(self.all_pths)

    def __getitem__(self, index):

        selected = self.all_pths[index]

        df = pd.read_csv(selected)

        x = df["x"].values.astype(np.float32)
        y = df["y"].values.astype(np.float32)

        u = df["u (m/s)"].values.astype(np.float32)
        v = df["v (m/s)"].values.astype(np.float32)
        P = df["p (Pa)"].values.astype(np.float32)

        # Original mask
        if "mask" in df.columns:
            original_mask = df["mask"].values.astype(np.float32)
        else:
            original_mask = np.ones_like(x, dtype=np.float32)

        # --------------------------------------------------
        # Normalize coordinates
        # --------------------------------------------------

        L = x.max() - x.min()

        x = (x - x.min()) / L

        height = (y.max() - y.min()) / L
        offset = (1 - height) / 2

        y = (y - y.min()) / L + offset

        # Points BEFORE filtering
        points = np.stack([x, y], axis=1)

        # --------------------------------------------------
        # Grid
        # --------------------------------------------------

        lin = np.linspace(0, 1, self.grid_size)
        grid_x, grid_y = np.meshgrid(lin, lin)

        # --------------------------------------------------
        # Create mask from ORIGINAL mask
        # --------------------------------------------------

        mask_grid = griddata(
            points,
            original_mask,
            (grid_x, grid_y),
            method="nearest",
            fill_value=0
        )

        mask = (mask_grid > 0.5).astype(np.float32)

        # --------------------------------------------------
        # Valid CFD points
        # --------------------------------------------------

        valid = (
            (original_mask > 0.5) &
            ~np.isnan(x) &
            ~np.isnan(y) &
            ~np.isnan(u) &
            ~np.isnan(v) &
            ~np.isnan(P)
        )

        x = x[valid]
        y = y[valid]
        u = u[valid]
        v = v[valid]
        P = P[valid]

        # IMPORTANT: recreate points after filtering
        points = np.stack([x, y], axis=1)

        # --------------------------------------------------
        # Interpolate u, v, P
        # --------------------------------------------------

        v_grid = griddata(
            points,
            v,
            (grid_x, grid_y),
            method="linear",
            fill_value=np.nan
        )

        u_grid = griddata(
            points,
            u,
            (grid_x, grid_y),
            method="linear",
            fill_value=np.nan
        )

        P_grid = griddata(
            points,
            P,
            (grid_x, grid_y),
            method="linear",
            fill_value=np.nan
        )

        # Replace NaNs
        u_grid = np.nan_to_num(u_grid, nan=0.0)
        v_grid = np.nan_to_num(v_grid, nan=0.0)
        P_grid = np.nan_to_num(P_grid, nan=0.0)

        # --------------------------------------------------
        # Clip
        # --------------------------------------------------

        u_grid[u_grid < config["Stats"][self.meta]["U_CLIP_MIN"]] = \
            config["Stats"][self.meta]["U_CLIP_MIN"]

        u_grid[u_grid > config["Stats"][self.meta]["U_CLIP_MAX"]] = \
            config["Stats"][self.meta]["U_CLIP_MAX"]

        v_grid[v_grid < config["Stats"][self.meta]["V_CLIP_MIN"]] = \
            config["Stats"][self.meta]["V_CLIP_MIN"]

        v_grid[v_grid > config["Stats"][self.meta]["V_CLIP_MAX"]] = \
            config["Stats"][self.meta]["V_CLIP_MAX"]

        P_grid[P_grid < config["Stats"][self.meta]["P_CLIP_MIN"]] = \
            config["Stats"][self.meta]["P_CLIP_MIN"]

        P_grid[P_grid > config["Stats"][self.meta]["P_CLIP_MAX"]] = \
            config["Stats"][self.meta]["P_CLIP_MAX"]

        # --------------------------------------------------
        # Boundary coordinates
        # --------------------------------------------------

        wall_xy, uv_xy = get_boundary_coordinates(
            mask,
            u_grid,
            v_grid,
            U=1.0,
            V=0.0
        )

        # --------------------------------------------------
        # Normalize velocity / pressure
        # --------------------------------------------------

        u_grid = (u_grid - config["Stats"][self.meta]["U_MEAN"]) / config["Stats"][self.meta]["U_STD"]

        v_grid = (v_grid - config["Stats"][self.meta]["V_MEAN"]) / config["Stats"][self.meta]["V_STD"]

        P_grid = (P_grid - config["Stats"][self.meta]["P_MEAN"]) / config["Stats"][self.meta]["P_STD"]

        uvp_grid = np.stack([u_grid, v_grid, P_grid], axis=0).astype(np.float32)

        number = float(selected.stem.split("_")[-1])

        return (
            uvp_grid,
            (number - config["Stats"][self.meta]["Re_Mean"]) / config["Stats"][self.meta]["Re_Std"],
            mask,
            wall_xy,
            uv_xy
        )

if __name__ == "__main__":

    dataset = dataset_csv(
        folder="Data/Problems/Backward_Facing_Step_domain",
        meta="Backward_Facing_Step"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True
    )

    uvp_grid, number, mask, wall_mask, uv_mask = next(iter(dataloader))

    # --------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------

    u = uvp_grid[0, 0].numpy()
    v = uvp_grid[0, 1].numpy()
    P = uvp_grid[0, 2].numpy()

    m = mask[0].numpy()

    wall_mask = wall_mask[0].numpy()
    uv_mask = uv_mask[0].numpy()

    # --------------------------------------------------
    # Print information
    # --------------------------------------------------

    print("u_min:", u.min())
    print("u_max:", u.max())

    print("v_min:", v.min())
    print("v_max:", v.max())

    print("P_min:", P.min())
    print("P_max:", P.max())

    print("uvp_grid shape:", uvp_grid.shape)
    print("number shape:", number.shape)
    print("mask shape:", mask.shape)

    print("wall_mask shape:", wall_mask.shape)
    print("uv_mask shape:", uv_mask.shape)

    print("wall points:", np.sum(wall_mask))
    print("uv points:", np.sum(uv_mask))

    # --------------------------------------------------
    # Grid
    # --------------------------------------------------

    x_grid = np.linspace(0, 1, 256)
    y_grid = np.linspace(0, 1, 256)

    fig, axes = plt.subplots(
        1, 4,
        figsize=(20, 5)
    )

    # ==================================================
    # U velocity
    # ==================================================

    cf = axes[0].contourf(
        x_grid,
        y_grid,
        u,
        levels=50,
        cmap="jet"
    )

    # Overlay wall BC
    wy, wx = np.where(wall_mask > 0.5)

    if len(wx) > 0:
        axes[0].scatter(
            x_grid[wx],
            y_grid[wy],
            s=8,
            marker="x",
            label="u=v=0"
        )

    # Overlay inlet BC
    uy, ux = np.where(uv_mask > 0.5)

    if len(ux) > 0:
        axes[0].scatter(
            x_grid[ux],
            y_grid[uy],
            s=12,
            marker="o",
            facecolors="none",
            label="u=U, v=V"
        )

    axes[0].set_title("U Velocity")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")

    if len(wx) > 0 or len(ux) > 0:
        axes[0].legend()

    fig.colorbar(cf, ax=axes[0])

    # ==================================================
    # V velocity
    # ==================================================

    cf = axes[1].contourf(
        x_grid,
        y_grid,
        v,
        levels=50,
        cmap="jet"
    )

    if len(wx) > 0:
        axes[1].scatter(
            x_grid[wx],
            y_grid[wy],
            s=8,
            marker="x"
        )

    if len(ux) > 0:
        axes[1].scatter(
            x_grid[ux],
            y_grid[uy],
            s=12,
            marker="o",
            facecolors="none"
        )

    axes[1].set_title("V Velocity")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")

    fig.colorbar(cf, ax=axes[1])

    # ==================================================
    # Pressure
    # ==================================================

    cf = axes[2].contourf(
        x_grid,
        y_grid,
        P,
        levels=50,
        cmap="jet"
    )

    axes[2].set_title("Pressure")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")

    fig.colorbar(cf, ax=axes[2])

    # ==================================================
    # Mask + Boundary Conditions
    # ==================================================

    cf = axes[3].contourf(
        x_grid,
        y_grid,
        m,
        levels=[-0.5, 0.5, 1.5],
        cmap="gray"
    )

    # Wall boundary
    if len(wx) > 0:
        axes[3].scatter(
            x_grid[wx],
            y_grid[wy],
            s=12,
            marker="x",
            label="u=v=0"
        )

    # Inlet boundary
    if len(ux) > 0:
        axes[3].scatter(
            x_grid[ux],
            y_grid[uy],
            s=20,
            marker="o",
            facecolors="none",
            label="u=U, v=V"
        )

    axes[3].set_title("Mask + Boundary Conditions")
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("y")
    axes[3].set_aspect("equal")

    if len(wx) > 0 or len(ux) > 0:
        axes[3].legend()

    # --------------------------------------------------
    # Overall formatting
    # --------------------------------------------------

    re_value = number.item()

    plt.suptitle(
        f"Backward Facing Step | Normalised Re = {re_value:.4f}",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()