"""Generate a SYNTHETIC wind timeseries in the format that
benchmarks/dei_layout.py `load_wind()` expects.

This is NOT real resource data — it is a reproducible synthetic stand-in so the
advanced discovery loop (`agent_cli.py --wind-csv`) and baseline/problem-JSON
(re)generation are runnable without redistributable measurements. The real
DEI/ROWP resources are not included.

Format (source of truth: benchmarks/dei_layout.py:172-185 `load_wind`):
  - delimiter ';', header row present
  - column WS_150 : wind speed at 150 m hub height, m/s
  - column WD_150 : wind direction, degrees in [0, 360)
  - leading (date) column is never parsed
The parser reduces the timeseries to a 24-bin wind rose, so only the empirical
(WD, WS) distribution matters — row count/timestep are cosmetic.

Usage:
  pixi run python scripts/make_sample_wind.py \
      [--out data/sample_wind_synthetic.csv] [--hours 8760] [--seed 0]
"""
import argparse
import datetime
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample_wind_synthetic.csv")
    ap.add_argument("--hours", type=int, default=8760, help="rows (1 year hourly)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    n = a.hours

    # Direction: prevailing south-west (~240 deg, like the North Sea site) via a
    # wrapped normal, mixed with a uniform background so the rose isn't flat.
    prevail = rng.normal(240.0, 35.0, n) % 360.0
    background = rng.uniform(0.0, 360.0, n)
    wd = np.where(rng.random(n) < 0.7, prevail, background) % 360.0

    # Speed: Weibull (k=2, scale A=11.9 -> mean ~10.5 m/s), with a weak
    # direction coupling (SW windier), floored at 1 m/s.
    ws = 11.9 * rng.weibull(2.0, n)
    ws = ws * (1.0 + 0.15 * np.cos(np.deg2rad(wd - 240.0)))
    ws = np.clip(ws, 1.0, None)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    start = datetime.datetime(2020, 1, 1)
    with open(a.out, "w") as f:
        f.write(";WS_150;WD_150\n")
        for i in range(n):
            ts = (start + datetime.timedelta(hours=i)).strftime("%Y-%m-%d %H:%M")
            f.write(f"{ts};{ws[i]:.6f};{wd[i]:.6f}\n")

    print(f"wrote {a.out}: {n} hourly rows, mean WS={ws.mean():.2f} m/s, "
          f"prevailing dir ~240 deg (SYNTHETIC, not real resource data)")


if __name__ == "__main__":
    main()
