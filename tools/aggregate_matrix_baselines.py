"""Aggregate per-seed baseline run outputs into a single matrix-level
JSON with per-cell best/mean AEP, feasibility rate, timing.

Reads lumi/logs/matrix_baselines/<farm>_n<N>_rose<rose>_seed<K>.out
files (each a JSON dict from run_single_baseline.py) and rolls up
to one entry per (farm, N, rose) cell in results/matrix/baselines_matrix.json.
"""
import argparse
import glob
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-dir", default="results/matrix")
    p.add_argument("--logs-dir", default="lumi/logs/matrix_baselines")
    p.add_argument("--out", default="results/matrix/baselines_matrix.json")
    args = p.parse_args()

    manifest = json.load(open(os.path.join(args.matrix_dir, "manifest.json")))

    cells_out = {}
    for cell in manifest["cells"]:
        key = f"{cell['farm']}_n{cell['n']}_rose{cell['rose']}"
        log_glob = os.path.join(args.logs_dir, f"{key}_seed*.out")
        paths = sorted(glob.glob(log_glob))
        aeps, feas = [], []
        times = []
        for p in paths:
            try:
                d = json.loads(open(p).read())
                if "aep" in d:
                    aeps.append(d["aep"])
                    feas.append(bool(d.get("feasible", False)))
                    times.append(d.get("time", 0.0))
            except Exception:
                continue
        if not aeps:
            cells_out[key] = {"error": "no data", "n_seeds": 0}
            continue
        # Feasibility-filtered best
        feas_aeps = [a for a, f in zip(aeps, feas) if f]
        best_aep = max(feas_aeps) if feas_aeps else max(aeps)
        cells_out[key] = {
            "farm": cell["farm"],
            "n": cell["n"],
            "rose": cell["rose"],
            "n_seeds": len(aeps),
            "best_aep": round(best_aep, 3),
            "mean_aep": round(sum(aeps) / len(aeps), 3),
            "feasible_rate": round(sum(feas) / len(aeps), 4),
            "total_time_s": round(sum(times), 1),
        }
        print(f"  {key}: {cells_out[key]['n_seeds']} seeds, "
              f"best={cells_out[key]['best_aep']}, "
              f"feas_rate={cells_out[key]['feasible_rate']}")

    with open(args.out, "w") as f:
        json.dump(cells_out, f, indent=2)
    n_filled = sum(1 for v in cells_out.values() if "best_aep" in v)
    print()
    print(f"Wrote {args.out} with {n_filled}/{len(cells_out)} cells filled")


if __name__ == "__main__":
    main()
