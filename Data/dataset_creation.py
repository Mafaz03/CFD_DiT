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

ROOT = Path(__file__).resolve().parent.parent

def create():

    parser = argparse.ArgumentParser(description="Dataset creation for multiple CFD comsol simulations")
    parser.add_argument("-f",   "--folder")
    parser.add_argument("-t",   "--threshold",     default = 0.01, type = float)
    parser.add_argument("-n",   "--grid_per_axis", default = 64,   type = int)
    parser.add_argument("-l_c", "--length_cap",    default = None, type = float)
    parser.add_argument("-s_p", "--shift_up",      default = 0.0,  type = float)

    args = parser.parse_args()

    files = os.listdir(f"{ROOT}/Data/Problems/{args.folder}")
    for file in tqdm(files):
        if file.endswith(".csv"):

            re = file.split("Re_")[-1].split(".csv")[0]

            df = pd.read_csv(f"{ROOT}/Data/Problems/{args.folder}/Re_{re}.csv")

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
            scale = x.max() - x.min()

            x_scaled = (x - x.min()) / scale
            y_scaled = (y - y.min()) / scale  

            xi = np.linspace(0, 1, args.grid_per_axis)
            yi = np.linspace(0, 1, args.grid_per_axis)

            X, Y = np.meshgrid(xi, yi)

            U = griddata((x_scaled, y_scaled+args.shift_up), u, (X, Y), method="cubic")
            V = griddata((x_scaled, y_scaled+args.shift_up), v, (X, Y), method="cubic")
            P = griddata((x_scaled, y_scaled+args.shift_up), p, (X, Y), method="cubic")

            # Build KD-tree from original points
            tree = cKDTree(np.c_[x_scaled, y_scaled + args.shift_up])

            # Distance from every grid point to nearest original point
            dist, _ = tree.query(np.c_[X.ravel(), Y.ravel()])

            # Reshape
            dist = dist.reshape(X.shape)

            # Mask points farther than a threshold
            mask = dist > args.threshold

            U = np.ma.masked_where(mask, U)
            V = np.ma.masked_where(mask, V)
            P = np.ma.masked_where(mask, P)

            mask = ~mask


            new_df = pd.DataFrame({
                                    "x": X.flatten(),
                                    "y": Y.flatten(),
                                    "u (m/s)": U.flatten(),
                                    "v (m/s)": V.flatten(),
                                    "p (Pa)" : P.flatten(),
                                    "mask"   : mask.flatten()
                                })
            
            os.makedirs(f"{ROOT}/Data/Problems/{args.folder}_domain", exist_ok=True)
            new_df.to_csv(f"{ROOT}/Data/Problems/{args.folder}_domain/Re_{re}.csv")

    fig, axes = plt.subplots(1, 4, figsize=(12, 5))
    a = axes[0].contourf(X, Y, U, levels=50, cmap="jet")
    plt.colorbar(a)
    axes[0].set_title("u velocity")

    a = axes[1].contourf(X, Y, V, levels=50, cmap="jet")
    axes[1].set_title("v velocity")
    plt.colorbar(a)

    a = axes[2].contourf(X, Y, P, levels=50, cmap="jet")
    axes[2].set_title("Pressure")
    plt.colorbar(a)

    a = axes[3].contourf(X, Y, mask, levels=50, cmap="jet")
    axes[3].set_title("Mask")
    plt.colorbar(a)

    plt.suptitle(f"Re (unnormalised): {re}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create()