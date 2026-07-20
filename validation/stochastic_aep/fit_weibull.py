"""Fit per-sector Weibull(A, k) to the DEI 10-year time-series wind resource.

Input: dependencies/pixwake/energy_island_10y_daily_av_wind.csv (~3653 daily-mean
samples of WS_150 and WD_150). Output: a JSON resource file with
  - n_sectors (12 — matches the published DEI convention)
  - sector_centers_deg, sector_width_deg
  - sector_probability (sums to 1)
  - weibull_A, weibull_k per sector
  - the original samples (for empirical-resampling validation)

Usage:
    pixi run python validation/stochastic_aep/fit_weibull.py \\
        dependencies/pixwake/energy_island_10y_daily_av_wind.csv \\
        validation/stochastic_aep/dei_weibull_12.json \\
        --n-sectors 12
"""
import argparse
import json
import sys

import numpy as np
from scipy.special import gamma
from scipy.stats import weibull_min


def fit_weibull_per_sector(ws, wd, n_sectors=12):
    """Fit Weibull A, k per direction sector using MLE.

    Returns:
        dict with sector_centers_deg, sector_width_deg, sector_probability,
        weibull_A, weibull_k (all length n_sectors), plus a global mean speed
        for sanity-check.
    """
    sector_width = 360.0 / n_sectors
    # Sector centers at 0, sector_width, 2*sector_width, ... (N convention)
    centers = np.arange(n_sectors) * sector_width

    # Bin each sample into a sector using nearest center modulo 360
    # A sample at wd belongs to sector i if (wd - centers[i]) mod 360 is
    # within ±sector_width/2.
    shifted = (wd + sector_width / 2) % 360.0  # rotate so sector 0 covers [0, w)
    sector_idx = (shifted // sector_width).astype(int) % n_sectors

    n_total = len(ws)
    A_arr = np.zeros(n_sectors)
    k_arr = np.zeros(n_sectors)
    freq_arr = np.zeros(n_sectors)

    for s in range(n_sectors):
        mask = sector_idx == s
        n_s = mask.sum()
        freq_arr[s] = n_s / n_total
        ws_s = ws[mask]
        if n_s < 2 or ws_s.std() == 0:
            # Degenerate fallback: assume Weibull with k=2.0, A from mean
            mean_s = ws_s.mean() if n_s > 0 else 0.0
            A_arr[s] = mean_s / gamma(1 + 1 / 2.0)
            k_arr[s] = 2.0
            continue
        # MLE fit with location=0
        k_hat, loc_hat, A_hat = weibull_min.fit(ws_s, floc=0)
        A_arr[s] = A_hat
        k_arr[s] = k_hat

    return {
        "n_sectors": n_sectors,
        "sector_centers_deg": list(map(float, centers)),
        "sector_width_deg": float(sector_width),
        "sector_probability": list(map(float, freq_arr)),
        "weibull_A": list(map(float, A_arr)),
        "weibull_k": list(map(float, k_arr)),
        "ws_mean_global": float(ws.mean()),
        "wd_mean_global": float(wd.mean()),
        "n_samples": int(n_total),
        "samples_ws": list(map(float, ws)),
        "samples_wd": list(map(float, wd)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("out_json")
    p.add_argument("--n-sectors", type=int, default=12)
    args = p.parse_args()

    # CSV has header ";WS_150;WD_150" and ; delimiter
    raw = np.loadtxt(args.csv, delimiter=";", skiprows=1, dtype=str)
    ws = raw[:, 1].astype(float)
    wd = raw[:, 2].astype(float)
    out = fit_weibull_per_sector(ws, wd, n_sectors=args.n_sectors)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    # Compact stdout summary
    summary = {
        "n_sectors": out["n_sectors"],
        "ws_mean": out["ws_mean_global"],
        "n_samples": out["n_samples"],
        "weibull_A": out["weibull_A"],
        "weibull_k": out["weibull_k"],
        "sector_probability": out["sector_probability"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
