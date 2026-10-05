import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
import os
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from tqdm import tqdm
import argparse
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
with open(f"{ROOT}/config/config.json", "r") as file:
    config = json.load(file)

def create():

    parser = argparse.ArgumentParser(description="Dataset creation for multiple CFD comsol simulations")
    parser.add_argument("-f",   "--folder")
    parser.add_argument("-m",   "--meta")
    parser.add_argument("-j",   "--json_file")
    parser.add_argument("-t",   "--threshold",     default = 0.01, type = float)
    parser.add_argument("-n",   "--grid_per_axis", default = 256,  type = int)
    parser.add_argument("-l_c", "--length_cap",    default = None, type = float)
    parser.add_argument("-s_p", "--shift_up",      default = 0.0,  type = float)

    args = parser.parse_args()

    with open(args.json_file, "r") as f:
        re_mapping = json.load(f)

    files = os.listdir(f"{ROOT}/{args.folder}")
    for file in tqdm(files):
        file = Path(file)
        if file.suffix.lower() == '.csv':

            re = float(re_mapping[file.stem])

            df = pd.read_csv(f"{ROOT}/{args.folder}/{file}")

            if args.length_cap:
                df = df[df["x"] <= args.length_cap]

            x = df["x"].values
            y = df["y"].values
            u = df["u (m/s)"].values
            v = df["v (m/s)"].values
            p = df["p (Pa)"].values

            valid = (
                ~np.isnan(x) &
                ~np.isnan(y) &
                ~np.isnan(u) &
                ~np.isnan(v) &
                ~np.isnan(p)
            )


            x = x[valid]
            y = y[valid]
            u = u[valid]
            v = v[valid]
            p = p[valid]

            # Scale factor so that x spans [0, 1]
            x_range = x.max() - x.min()
            y_range = y.max() - y.min()

            fits = (x.min() >= 0 and x.max() <= 1 and
                    y.min() >= 0 and y.max() <= 1)

            if fits:
                # Already inside the canvas: keep the original size
                x_scaled = x.copy()
                y_scaled = y.copy()
            else:
                # Outside [0, 1]: scale by the longer side, preserving aspect ratio
                scale = max(x_range, y_range)
                x_scaled = (x - x.min()) / scale
                y_scaled = (y - y.min()) / scale

            x_scaled = x_scaled - x_scaled.min() 
            y_scaled = y_scaled - y_scaled.min() 

            xi = np.linspace(0, 1, args.grid_per_axis)
            yi = np.linspace(0, 1, args.grid_per_axis)

            X, Y = np.meshgrid(xi, yi)


            U = griddata(
                (x_scaled, y_scaled), u, (X, Y),
                method="linear",
                fill_value=np.nan
            )

            V = griddata(
                (x_scaled, y_scaled), v, (X, Y),
                method="linear",
                fill_value=np.nan
            )

            P = griddata(
                (x_scaled, y_scaled), p, (X, Y),
                method="linear",
                fill_value=np.nan
            )

            # Valid CFD region
            mask = (
                ~np.isnan(U) &
                ~np.isnan(V) &
                ~np.isnan(P)
            )

            # Keep NaN outside the valid region
            U = np.where(mask, U, np.nan)
            V = np.where(mask, V, np.nan)
            P = np.where(mask, P, np.nan)

            # Clip only valid values
            U[mask & (U < config["Stats"][args.meta]["U_CLIP_MIN"])] = config["Stats"][args.meta]["U_CLIP_MIN"]
            U[mask & (U > config["Stats"][args.meta]["U_CLIP_MAX"])] = config["Stats"][args.meta]["U_CLIP_MAX"]

            V[mask & (V < config["Stats"][args.meta]["V_CLIP_MIN"])] = config["Stats"][args.meta]["V_CLIP_MIN"]
            V[mask & (V > config["Stats"][args.meta]["V_CLIP_MAX"])] = config["Stats"][args.meta]["V_CLIP_MAX"]

            P[mask & (P < config["Stats"][args.meta]["P_CLIP_MIN"])] = config["Stats"][args.meta]["P_CLIP_MIN"]
            P[mask & (P > config["Stats"][args.meta]["P_CLIP_MAX"])] = config["Stats"][args.meta]["P_CLIP_MAX"]

            new_df = pd.DataFrame({
                "x": X.flatten(),
                "y": Y.flatten(),
                "u (m/s)": U.flatten(),
                "v (m/s)": V.flatten(),
                "p (Pa)": P.flatten(),
                "mask": mask.flatten()
            })
            os.makedirs(f"{ROOT}/{args.folder}_domain", exist_ok=True)
            new_df.to_csv(f"{ROOT}/{args.folder}_domain/{file.stem}.csv")

    fig, axes = plt.subplots(1, 5, figsize=(12, 5))
    a = axes[0].contourf(X, Y, U, levels=50, cmap="jet")
    plt.colorbar(a)
    axes[0].set_title("u velocity")

    a = axes[1].contourf(X, Y, V, levels=50, cmap="jet")
    axes[1].set_title("v velocity")
    plt.colorbar(a)

    a = axes[2].contourf(X, Y, ((U**2) + (V**2))**0.5, levels=50, cmap="jet")
    axes[1].set_title("mag")
    plt.colorbar(a)

    a = axes[3].contourf(X, Y, P, levels=50, cmap="jet")
    axes[2].set_title("Pressure")
    plt.colorbar(a)

    a = axes[4].contourf(X, Y, mask, levels=50, cmap="jet")
    axes[3].set_title("Mask")
    plt.colorbar(a)

    plt.suptitle(f"Re (unnormalised): {re}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create()