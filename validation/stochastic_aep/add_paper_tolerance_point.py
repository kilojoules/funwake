"""Add the exact paper tolerance point ηT = 0.1 m to the δ-sweep on the 4
low-margin cells. Corresponds to δ = ηT / η₀ = 0.1 / 50 = 0.002.

Runs 4 cells × 3 sample seeds × decay+ES baseline at δ = 0.002.
"""
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from run_hardening import LOW_MARGIN_CELLS, run_task


def main():
    out_path = "/Users/julianquick/portfolio_copy/funwake/validation/stochastic_aep/eta_t_paper_point.json"
    if os.path.exists(out_path):
        results = json.load(open(out_path))
    else:
        results = {"runs": []}
    seen = {(r["cell_path"], r["sample_seed"], r.get("delta")) for r in results["runs"]}

    tasks = []
    for cell in LOW_MARGIN_CELLS:
        for ss in (100000, 200000, 300000):
            if (cell, ss, 0.002) in seen:
                continue
            tasks.append({
                "kind": "h1_refined_delta", "cell_path": cell,
                "delta": 0.002, "sample_seed": ss, "init_seed": 0,
            })

    print(f"Tasks: {len(tasks)} (paper ηT=0.1m point, δ=0.002, 4 low-margin cells)", flush=True)
    t_start = time.time()
    for i, t in enumerate(tasks, 1):
        t0 = time.time()
        r = run_task(t)
        results["runs"].append(r)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        tag = ("ERR " + r["error"][:80]) if "error" in r else (
            f"AEP={r['aep_det_gwh']:.2f} bp_final={r['bp_final']:.1e} elapsed={r['elapsed_s']}s"
        )
        print(f"[{i}/{len(tasks)}] {os.path.basename(r['cell_path']):42s} ss={r['sample_seed']}  {tag}",
              flush=True)
    print(f"\nWall: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
