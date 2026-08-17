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
                 grid_size: int = 256):

        self.folder = Path(folder)
        allowed_exts = {".csv"}

        self.all_pths = [
            self.folder / name
            for name in os.listdir(self.folder)
            if Path(name).suffix.lower() in allowed_exts
        ]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])

        self.grid_size = grid_size

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

        u = (u - config["Stats"]["U_MEAN"])/config["Stats"]["U_STD"]
        v = (v - config["Stats"]["V_MEAN"])/config["Stats"]["V_STD"]
        P = (P - config["Stats"]["P_MEAN"])/config["Stats"]["P_STD"]

        L = x.max() - x.min()
        x = (x - x.min()) / L
        height = (y.max() - y.min()) / L
        offset = (1 - height)/2
        y = (y - y.min()) / L + offset

        points = np.stack([x, y], axis=1)  # (N, 3)

        # interpolate u, v, P onto regular grid
        # regular grid to interpolate onto
        lin = np.linspace(0, 1, self.grid_size)
        grid_x, grid_y = np.meshgrid(lin, lin)  # (grid_size, grid_size)

        v_grid = griddata(points, v, (grid_x, grid_y), method="linear", fill_value = np.nan)
        u_grid = griddata(points, u, (grid_x, grid_y), method="linear", fill_value = np.nan)
        P_grid = griddata(points, P, (grid_x, grid_y), method="linear", fill_value = np.nan)

        mask = ~np.isnan(u_grid)          # (H,W)
        mask = mask.astype(np.float32)

        # replacing NaN with 0 but remembering NaN in mask
        u_grid = np.nan_to_num(u_grid, nan=0.0)
        v_grid = np.nan_to_num(v_grid, nan=0.0)
        P_grid = np.nan_to_num(P_grid, nan=0.0)

        uvp_grid = np.stack([u_grid, v_grid, P_grid], axis=0).astype(np.float32)  # (3, 256, 256)

        number = float(selected.stem.split("_")[-1])

        return uvp_grid, (number - config["Stats"]["Re_Mean"])/config["Stats"]["Re_Std"], mask


if __name__ == "__main__":
    dataset     = dataset_csv(folder = "Data/Problems/Backward_Facing_Step_domain")
    dataloader  = DataLoader(dataset, batch_size = 1, shuffle = True)

    uvp_grid, number, mask = next(iter(dataloader))

    u = uvp_grid[0][0, :, :]                                                 # (grid_size, grid_size)
    v = uvp_grid[0][1, :, :]                                                 # (grid_size, grid_size)
    P = uvp_grid[0][2, :, :]                                                 # (grid_size, grid_size)
    m = mask[0]

    print("u_min", u.min())
    print("u_max", u.max())
    print("u_mean", u.mean())
    print("u_std", u.std())

    print("v_min", v.min())
    print("v_max", v.max())
    print("v_mean", v.mean())
    print("v_std", v.std())

    print("P_min", P.min())
    print("P_max", P.max())
    print("P_mean", P.mean())
    print("P_std", P.std())
    
    print(uvp_grid.shape, number.shape, mask.shape)

    ### plotting ###

    

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
    
