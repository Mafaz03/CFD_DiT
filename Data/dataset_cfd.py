import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from torchvision import transforms
import numpy as np

from scipy.interpolate import griddata

import json

ROOT = Path(__file__).resolve().parent.parent

with open(f"{ROOT}/config/config.json", "r") as file:
    config = json.load(file)


class dataset_csv(Dataset):
    def __init__(self, folder: str,
                 meta: str,
                 grid_size: int = 256):

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

        # drop rows flagged untrustworthy upstream (create_dataset.py's, mask column), plus defensively drop any remaining NaNs
        # this second (64x64 -> 256x256) interpolation pass.
        if "mask" in df.columns:
            valid = df["mask"].values.astype(bool)
        else:
            valid = np.ones_like(x, dtype=bool)

        valid = valid & ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(u) & ~np.isnan(v) & ~np.isnan(P)

        x, y, u, v, P = x[valid], y[valid], u[valid], v[valid], P[valid]

        L = x.max() - x.min()
        x = (x - x.min()) / L
        height = (y.max() - y.min()) / L
        offset = (1 - height) / 2
        y = (y - y.min()) / L + offset

        points = np.stack([x, y], axis=1)  # (N, 2)

        lin = np.linspace(0, 1, self.grid_size)
        grid_x, grid_y = np.meshgrid(lin, lin)  # (grid_size, grid_size)

        v_grid = griddata(points, v, (grid_x, grid_y), method="linear", fill_value=np.nan)
        u_grid = griddata(points, u, (grid_x, grid_y), method="linear", fill_value=np.nan)
        P_grid = griddata(points, P, (grid_x, grid_y), method="linear", fill_value=np.nan)

        mask = ~(np.isnan(u_grid) | np.isnan(v_grid) | np.isnan(P_grid))  # (H,W)
        mask = mask.astype(np.float32)

        # replacing NaN with 0 but remembering NaN in mask
        u_grid = np.nan_to_num(u_grid, nan=0.0)
        v_grid = np.nan_to_num(v_grid, nan=0.0)
        P_grid = np.nan_to_num(P_grid, nan=0.0)

        # taking care of extreme values
        u_grid[u_grid < config["Stats"][self.meta]["U_CLIP_MIN"]] = config["Stats"][self.meta]["U_CLIP_MIN"]
        u_grid[u_grid > config["Stats"][self.meta]["U_CLIP_MAX"]] = config["Stats"][self.meta]["U_CLIP_MAX"]

        v_grid[v_grid < config["Stats"][self.meta]["V_CLIP_MIN"]] = config["Stats"][self.meta]["V_CLIP_MIN"]
        v_grid[v_grid > config["Stats"][self.meta]["V_CLIP_MAX"]] = config["Stats"][self.meta]["V_CLIP_MAX"]

        P_grid[P_grid < config["Stats"][self.meta]["P_CLIP_MIN"]] = config["Stats"][self.meta]["P_CLIP_MIN"]
        P_grid[P_grid > config["Stats"][self.meta]["P_CLIP_MAX"]] = config["Stats"][self.meta]["P_CLIP_MAX"]


        # normalising
        u_grid = (u_grid - config["Stats"][self.meta]["U_MEAN"]) / config["Stats"][self.meta]["U_STD"]
        v_grid = (v_grid - config["Stats"][self.meta]["V_MEAN"]) / config["Stats"][self.meta]["V_STD"]
        P_grid = (P_grid - config["Stats"][self.meta]["P_MEAN"]) / config["Stats"][self.meta]["P_STD"]


        uvp_grid = np.stack([u_grid, v_grid, P_grid], axis=0).astype(np.float32)  # (3, 256, 256)

        number = float(selected.stem.split("_")[-1])

        return uvp_grid, (number - config["Stats"][self.meta]["Re_Mean"]) / config["Stats"][self.meta]["Re_Std"], mask


if __name__ == "__main__":

    dataset = dataset_csv(folder="Data/Problems/Lid_Driven_domain", meta = "Lid_Driven")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    uvp_grid, number, mask = next(iter(dataloader))

    u = uvp_grid[0][0, :, :]
    v = uvp_grid[0][1, :, :]
    P = uvp_grid[0][2, :, :]
    m = mask[0]

    print("u_min", u.min())
    print("u_max", u.max())

    print("v_min", v.min())
    print("v_max", v.max())

    print("P_min", P.min())
    print("P_max", P.max())

    print(uvp_grid.shape, number.shape, mask.shape)

    x_grid = torch.linspace(0, 1, 256)
    y_grid = torch.linspace(0, 1, 256)

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))

    cf = axes[0].contourf(x_grid, y_grid, u, levels=50, cmap="jet")
    axes[0].set_title("u velocity")
    fig.colorbar(cf, ax=axes[0])

    cf = axes[1].contourf(x_grid, y_grid, v, levels=50, cmap="jet")
    axes[1].set_title("v velocity")
    fig.colorbar(cf, ax=axes[1])

    cf = axes[2].contourf(x_grid, y_grid, P, levels=50, cmap="jet")
    axes[2].set_title("Pressure")
    fig.colorbar(cf, ax=axes[2])

    cf = axes[3].contourf(x_grid, y_grid, m, levels=50, cmap="jet")
    axes[3].set_title("Mask")
    fig.colorbar(cf, ax=axes[3])

    plt.suptitle(f"Re (normalised): {number.item()}")
    plt.tight_layout()
    plt.show()