#!/usr/bin/env python
"""Aggregate the checkpointed evaluator results into G1/G2/G5 PASS/FAIL tables.

Reads funwake2/state/gates/*.json (one per tag/cell/steps/seed) and reports
per-(tag,cell,steps) feasible-mean AEP, feasibility rate, std, SEM. For G1
(native @6000) it compares to the lr0_diameter_rule baseline within the noise
floor; G2 (native @8000) is flagged as the IN-SEARCH c*D baseline.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FW2 = os.path.dirname(HERE)
GATES = os.path.join(FW2, "state", "gates")

# lr0_diameter_rule baseline means (results/lr0_diameter_rule/aggregate.json)
BASELINE = {
    "dei_n50": (5561.34, 0.65, 1.0),     # (mean, sem, noise_floor_GWh)
    "rowp_n74": (4261.72, 0.83, 1.0),
    "parque_n20": (231.06, 0.32, 0.5),
}
BASE_FEAS = {"dei_n50": 1.0, "rowp_n74": 1.0, "parque_n20": 0.9}


def load():
    recs = {}
    for f in glob.glob(os.path.join(GATES, "*.json")):
        d = json.load(open(f))
        key = (d.get("tag"), d.get("cell"), d.get("steps"), d.get("gamma_min"))
        recs.setdefault(key, []).append(d)
    return recs


def summarize(rows):
    ok = [r for r in rows if "aep_gwh" in r and not r.get("error")]
    feas = [r for r in ok if r.get("feasible")]
    aeps = np.array([r["aep_gwh"] for r in feas]) if feas else np.array([])
    return {
        "n": len(rows), "n_ok": len(ok), "n_feas": len(feas),
        "feas_rate": round(len(feas) / len(ok), 3) if ok else None,
        "mean": round(float(aeps.mean()), 3) if len(aeps) else None,
        "std": round(float(aeps.std()), 3) if len(aeps) else None,
        "sem": round(float(aeps.std() / np.sqrt(len(aeps))), 3) if len(aeps) else None,
    }


def main():
    recs = load()
    print("=== G1/G2/G5 aggregation ===")
    g1_pass = True
    g1_rows, g2_rows = [], []
    for (tag, cell, steps, gm), rows in sorted(recs.items(),
                                               key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        s = summarize(rows)
        line = (f"  {tag:8s} {cell:11s} steps={steps} gm={gm}: "
                f"feas={s['n_feas']}/{s['n_ok']} mean={s['mean']} "
                f"std={s['std']} sem={s['sem']}")
        # G1 check
        if tag == "native" and steps == 6000 and cell in BASELINE:
            tgt, tsem, floor = BASELINE[cell]
            if s["mean"] is not None:
                delta = round(s["mean"] - tgt, 3)
                thr = max(floor, 2 * (tsem + (s["sem"] or 0)))
                ok = abs(delta) <= thr and (s["feas_rate"] or 0) >= BASE_FEAS[cell] - 1e-9
                g1_pass = g1_pass and ok
                line += (f"  | G1 target={tgt} delta={delta:+.3f} "
                         f"thr=+-{thr:.2f} base_feas>={BASE_FEAS[cell]} "
                         f"-> {'PASS' if ok else 'FAIL'}")
                g1_rows.append((cell, s, tgt, delta, ok))
        if tag == "native" and steps == 8000 and cell in BASELINE:
            line += "  | G2 IN-SEARCH cD baseline"
            g2_rows.append((cell, s))
        print(line)

    print(f"\nG1 (native fidelity @6000, go/no-go): "
          f"{'PASS' if g1_pass and g1_rows else 'INCOMPLETE/FAIL'} "
          f"({len(g1_rows)}/3 cells checked)")
    print("G2 8000-step IN-SEARCH baselines:")
    for cell, s in g2_rows:
        print(f"   {cell}: mean={s['mean']} std={s['std']} sem={s['sem']} "
              f"feas={s['n_feas']}/{s['n_ok']}")


if __name__ == "__main__":
    main()
