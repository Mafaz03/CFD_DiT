from pathlib import Path
import sys
import os
sys.path.append(str(Path().resolve().parent))
str(Path().resolve().parent)

ROOT = Path(os.getcwd()).resolve().parent


import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pandas as pd
from scipy.interpolate import griddata
import matplotlib.ticker as tkr

from DDPM import LinearNoiseScheduler
from DiT_model import DiT
from VAE import VAE




def ground_truth(data_folder: str, re: float, plot: bool):
    assert f"Re_{re}.csv" in os.listdir(f"{ROOT}/Data/Problems/{data_folder}"), f"{re} make sure re is in the csv folder"
    df = pd.read_csv(f"{ROOT}/Data/Problems/{data_folder}/Re_{re}.csv", index_col = 0)

    lin = np.linspace(0, 1, 256)

    df = df.dropna(subset=["x", "y", "u (m/s)", "v (m/s)", "p (Pa)"])

    x = df['x']
    y = df['y']
    u = df["u (m/s)"].values.astype(np.float32)
    v = df["v (m/s)"].values.astype(np.float32)
    P = df["p (Pa)"].values.astype(np.float32)

    points = np.stack([x, y], axis=1)

    grid_x, grid_y = np.meshgrid(lin, lin)

    u_grid = griddata(points, u, (grid_x, grid_y), method="linear", fill_value=0.0)
    v_grid = griddata(points, v, (grid_x, grid_y), method="linear", fill_value=0.0)
    P_grid = griddata(points, P, (grid_x, grid_y), method="linear", fill_value=0.0)

    dx = lin[1] - lin[0]
    dy = lin[1] - lin[0]

    du_dx = np.gradient(u_grid, dx, axis = 1)
    du_dy = np.gradient(u_grid, dy, axis = 0)
    dv_dy = np.gradient(v_grid, dy, axis = 0)
    dv_dx = np.gradient(v_grid, dx, axis = 1)

    d2u_dx2 = np.gradient(du_dx, dx, axis = 1)
    d2u_dy2 = np.gradient(du_dy, dy, axis = 0)
    d2v_dy2 = np.gradient(dv_dy, dy, axis = 0)
    d2v_dx2 = np.gradient(dv_dx, dx, axis = 1)

    dP_dx = np.gradient(P_grid, dx, axis = 1)
    dP_dy = np.gradient(P_grid, dy, axis = 0)

    div = du_dx + dv_dy

    momentum_res_x = (1 * ((u_grid * du_dx) + (v_grid * du_dy))) + dP_dx - (1/re) * (d2u_dx2 + d2u_dy2)
    momentum_res_y = (1 * ((u_grid * dv_dx) + (v_grid * dv_dy))) + dP_dy - (1/re) * (d2v_dx2 + d2v_dy2)

    momentum_res = (momentum_res_x ** 2 + momentum_res_y ** 2) ** 0.5

    mag = np.sqrt(u_grid**2 + v_grid**2)

    if plot:
        fig, axes = plt.subplots(1, 6, figsize=(30, 5))

        u_c = axes[0].contourf(u_grid, levels=300, cmap="jet")
        plt.colorbar(u_c, ax=axes[0])
        axes[0].set_title("u")

        v_c = axes[1].contourf(v_grid, levels=300, cmap="jet")
        plt.colorbar(v_c, ax=axes[1])
        axes[1].set_title("v")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")

        p_c = axes[2].contourf(P_grid, levels=300, cmap="jet")
        plt.colorbar(p_c, ax=axes[2])
        axes[2].set_title("P")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")

        mag_c = axes[3].contourf(mag, levels=300, cmap="jet")
        plt.colorbar(mag_c, ax=axes[3])
        axes[3].set_title("Magnitude")
        axes[3].set_xlabel("x")
        axes[3].set_ylabel("y")

        div_c = axes[4].contourf(grid_x, grid_y, div, levels=300, cmap="seismic")
        plt.colorbar(div_c, ax=axes[4])
        axes[4].set_title(f"Divergence mean: {div.mean():.4f}")
        axes[4].set_xlabel("x")
        axes[4].set_ylabel("y")

        div_m = axes[5].contourf(grid_x, grid_y, momentum_res, levels=300, cmap="seismic")
        plt.colorbar(div_m, ax=axes[5])
        axes[5].set_title(f"Momentum residual mean: {momentum_res.mean():.4f}")
        axes[5].set_xlabel("x")
        axes[5].set_ylabel("y")

        plt.tight_layout()
        plt.show()

    return u_grid, v_grid, P_grid, div, momentum_res, mag




def to_unit_range(x, lo, hi, **kwargs):   return 2.0 * (x - lo) / (hi - lo) - 1.0
def from_unit_range(y, lo, hi, **kwargs): return (y + 1.0) / 2.0 * (hi - lo) + lo


def predict(dit, vae, scheduler,
            re_value, sample_fn, 
            re_mean, re_std, 
            u_clip_min, u_clip_max, 
            v_clip_min, v_clip_max, 
            P_clip_min, P_clip_max, 
            grid_size,
            device="cpu", plot: bool = True, **kwargs):
    
    '''
    magic
    '''

    img = sample_fn(latent_grid_size = 32, 
                    dit              = dit, 
                    vae              = vae,
                    device           = device, 
                    scheduler        = scheduler,
                    number           = (re_value - re_mean) / re_std, 
                    **kwargs)

    boundaries = kwargs.get("boundaries", None)

    img_np = img.detach().cpu().numpy() if hasattr(img, "detach") else np.array(img)

    img_np = img_np[0]

    u = from_unit_range(img_np[0], u_clip_min, u_clip_max, eps = 0.001)
    v = from_unit_range(img_np[1], v_clip_min, v_clip_max, eps = 0.001)
    p = from_unit_range(img_np[2], P_clip_min, P_clip_max, eps = 0.001)

    mag = np.sqrt(u**2 + v**2)

    dx = 1.0 / (grid_size - 1)  # grid spacing over [0, 1]

    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dx, axis=0)
    dv_dy = np.gradient(v, dx, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)

    d2u_dx2 = np.gradient(du_dx, dx, axis=1)
    d2u_dy2 = np.gradient(du_dy, dx, axis=0)
    d2v_dy2 = np.gradient(dv_dy, dx, axis=0)
    d2v_dx2 = np.gradient(dv_dx, dx, axis=1)

    dP_dx = np.gradient(p, dx, axis=1)
    dP_dy = np.gradient(p, dx, axis=0)

    momentum_res_x = ((u * du_dx) + (v * du_dy)) + dP_dx - (1/re_value) * (d2u_dx2 + d2u_dy2)
    momentum_res_y = ((u * dv_dx) + (v * dv_dy)) + dP_dy - (1/re_value) * (d2v_dx2 + d2v_dy2)

    momentum_res = (momentum_res_x ** 2 + momentum_res_y ** 2) ** 0.5

    divergence = du_dx + dv_dy

    if plot:
        x_grid = np.linspace(0, 1, grid_size)
        y_grid = np.linspace(0, 1, grid_size)

        fig, axes = plt.subplots(1, 6, figsize=(30, 5))

        c0 = axes[0].contourf(x_grid, y_grid, np.where(boundaries.squeeze(0), u, np.nan), levels=200, cmap="jet")
        fig.colorbar(c0, ax=axes[0])
        axes[0].set_title("u (m/s)")

        c1 = axes[1].contourf(x_grid, y_grid, np.where(boundaries.squeeze(0), v, np.nan), levels=200, cmap="jet")
        fig.colorbar(c1, ax=axes[1])
        axes[1].set_title("v (m/s)")

        c2 = axes[2].contourf(x_grid, y_grid, np.where(boundaries.squeeze(0), p, np.nan), levels=200, cmap="jet")
        fig.colorbar(c2, ax=axes[2])
        axes[2].set_title("P (Pa)")

        c3 = axes[3].contourf(x_grid, y_grid, np.where(boundaries.squeeze(0), mag, np.nan), levels=200, cmap="jet")
        fig.colorbar(c3, ax=axes[3])
        axes[3].set_title("|U|")

        c4 = axes[4].contourf(x_grid, y_grid, np.where(boundaries.squeeze(0), divergence, np.nan), levels=200, cmap="seismic")
        fig.colorbar(c4, ax=axes[4], label="∇·u")
        axes[4].set_title(f"divergence mean: {divergence.mean():.2f}")

        div_m = axes[5].contourf(x_grid, y_grid, np.where(boundaries.squeeze(0), momentum_res, np.nan), levels=200, cmap="seismic")
        plt.colorbar(div_m, ax=axes[5])
        axes[5].set_title(f"Momentum residual mean: {momentum_res.mean():.4f}")
        axes[5].set_xlabel("x")
        axes[5].set_ylabel("y")

        plt.suptitle(f"Re = {re_value}")
        plt.tight_layout()
        plt.show()

    return u, v, p, divergence, momentum_res, mag



