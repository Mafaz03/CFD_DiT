import pandas as pd
import numpy as np
import re
from pathlib import Path
import json
from scipy.interpolate import griddata


# ============================================================
# COMSOL DAT FILES -> CSV CONVERTION
# ============================================================

file_paths = ["Data/re_full.dat", "Data/re_06_hori.dat", "Data/re_06_verti.dat", "Data/re_rand.dat"]
output_dirs = [Path("Data/Problems/re_full"), Path("Data/Problems/re_06_hori"), Path("Data/Problems/re_06_verti"), Path("Data/Problems/re_rand")]

# file_paths = ["Data/re_06_verti.dat"]
# output_dirs = [Path("Data/re_06_verti")]


for file_path, output_dir in zip(file_paths, output_dirs):
    comment_chars = ("%", "#", "!", "/", ";", "@")

    skip = 0
    lines = []

    with open(file_path, "r") as fh:

        for line in fh:

            stripped = line.strip()

            if stripped and stripped[0] in comment_chars:

                skip += 1

                if stripped.startswith("% x,y"):

                    print("header found")

                    stripped = stripped[2:]

                    # Header is comma separated
                    lines.append(stripped.split(","))

            elif stripped:

                # Numerical data is comma separated
                values = stripped.split(",")
                values = [float(v) for v in values]

                lines.append(values)


    if skip:
        print(f"Skipping {skip} header line(s)")


    df = pd.DataFrame(lines)

    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)

    print("Original COMSOL data:", df.shape)


    re_values = []

    for col in df.columns:

        match = re.search(r"Re=([0-9.]+)", str(col))

        if match:

            Re = match.group(1)

            if Re not in re_values:
                re_values.append(Re)

    print("Number of Reynolds numbers:", len(re_values))



    x = df["x"].astype(float).values
    y = df["y"].astype(float).values

    N = 256

    # Preserve the original aspect ratio
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    scale = max(x_range, y_range)

    x = (x - x.min()) / scale
    y = (y - y.min()) / scale

    # Center the smaller dimension inside the [0, 1] square
    x += (1 - x_range / scale) / 2
    y += (1 - y_range / scale) / 2

    xi = np.linspace(0, 1, N)
    yi = np.linspace(0, 1, N)

    X, Y = np.meshgrid(xi, yi)

    points = np.column_stack((x, y))

    output_dir.mkdir(exist_ok=True)

    re_json = {}

    for i in range(2, len(df.columns), 3):

        u_col = df.columns[i]
        v_col = df.columns[i + 1]
        p_col = df.columns[i + 2]

        # Extract Re from actual column name
        match = re.search(r"Re=([0-9.]+)", str(u_col))

        if match is None:
            raise ValueError(f"Could not find Re in {u_col}")

        Re = match.group(1)

        print(f"Processing Re = {Re}")

        u = df[u_col].astype(float).values
        v = df[v_col].astype(float).values
        p = df[p_col].astype(float).values


        U = griddata(points, u, (X, Y), method="linear")

        V = griddata(points, v, (X, Y), method="linear")

        P = griddata(points, p, (X, Y), method="linear")




        output = pd.DataFrame({
            "x": X.ravel(),
            "y": Y.ravel(),
            "u (m/s)": U.ravel(),
            "v (m/s)": V.ravel(),
            "p (Pa)": P.ravel()
        })


        filename = f"Re_{Re}.csv"

        output.to_csv(
            output_dir / filename,
            index=False
        )

        re_json[filename] = float(Re)

        print(f"Saved {filename}")



    with open(output_dir / "re.json", "w") as f:
        json.dump(re_json, f, indent=4)

    print("Done.")
