# preprocessing the stats for z score normalization; and clipping the outliers based on percentile 
# griddata/noise/outliers can create sudden spikes that can throw everything on the wrong track
# to fix this we use clipping;

## NOTE that 'CLIP' is different from openai's CLIP, here it means clipping the values

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import argparse

ROOT = Path(__file__).resolve().parent.parent


def compute_raw_stats(folder: str, meta: str, low_pct: float = 0.5, high_pct: float = 99.5,
                       length_cap: float = None):
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".csv")

    all_u, all_v, all_p, all_re = [], [], [], []

    for f in tqdm(files, desc=f"[{meta}] reading raw CSVs"):
        df = pd.read_csv(f)

        if length_cap:
            df = df[df["x"] <= length_cap]

        u = df["u (m/s)"].values.astype(np.float64)
        v = df["v (m/s)"].values.astype(np.float64)
        p = df["p (Pa)"].values.astype(np.float64)

        valid = ~np.isnan(u) & ~np.isnan(v) & ~np.isnan(p)
        u, v, p = u[valid], v[valid], p[valid]

        all_u.append(u)
        all_v.append(v)
        all_p.append(p)

        re_value = float(f.stem.split("Re_")[-1])
        all_re.append(re_value)

    all_u = np.concatenate(all_u)
    all_v = np.concatenate(all_v)
    all_p = np.concatenate(all_p)
    all_re = np.array(all_re)

    # robust percentile-based clip bounds -- protects against a single
    # corrupted row/solver artifact setting the bound for the whole dataset
    u_clip_min, u_clip_max = np.percentile(all_u, [low_pct, high_pct])
    v_clip_min, v_clip_max = np.percentile(all_v, [low_pct, high_pct])
    p_clip_min, p_clip_max = np.percentile(all_p, [low_pct, high_pct])

    # mean/std computed AFTER clipping, so they match the distribution
    # training will actually see once dataset_creation.py applies the
    # same clip bounds
    u_clipped = np.clip(all_u, u_clip_min, u_clip_max)
    v_clipped = np.clip(all_v, v_clip_min, v_clip_max)
    p_clipped = np.clip(all_p, p_clip_min, p_clip_max)

    stats = {
        "Re_Mean": float(all_re.mean()), "Re_Std": float(all_re.std()),
        "Re_Min": float(all_re.min()),   "Re_Max": float(all_re.max()),

        "U_MEAN": float(u_clipped.mean()), "U_STD": float(u_clipped.std()),
        "U_CLIP_MIN": float(u_clip_min),   "U_CLIP_MAX": float(u_clip_max),

        "V_MEAN": float(v_clipped.mean()), "V_STD": float(v_clipped.std()),
        "V_CLIP_MIN": float(v_clip_min),   "V_CLIP_MAX": float(v_clip_max),

        "P_MEAN": float(p_clipped.mean()), "P_STD": float(p_clipped.std()),
        "P_CLIP_MIN": float(p_clip_min),   "P_CLIP_MAX": float(p_clip_max),
    }

    print(f"\n--- Stats for meta='{meta}' ---")
    for k, v in stats.items():
        print(f"{k}: {v:.6f}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute raw per-meta CFD stats + clip bounds")
    parser.add_argument("-f", "--folder", required=True,
                         help="Path to the RAW source CSV folder (e.g. Data/Problems/Lid_Driven_domain)")
    parser.add_argument("-m", "--meta", required=True,
                         help="Meta/problem-domain key, e.g. 'Lid_Driven' or 'Flow_Past_Cylinder'")
    parser.add_argument("-lo", "--low_pct", default=0.5, type=float)
    parser.add_argument("-hi", "--high_pct", default=99.5, type=float)
    parser.add_argument("-l_c", "--length_cap", default=None, type=float)
    args = parser.parse_args()

    stats = compute_raw_stats(args.folder, args.meta, args.low_pct, args.high_pct, args.length_cap)

    # Write/merge into config.json automatically under Stats[meta]
    config_path = f"{ROOT}/config/config.json"
    with open(config_path, "r") as file:
        config = json.load(file)

    config.setdefault("Stats", {})
    config["Stats"][args.meta] = stats

    with open(config_path, "w") as file:
        json.dump(config, file, indent=2)

    print(f"\nWrote Stats['{args.meta}'] into {config_path}")