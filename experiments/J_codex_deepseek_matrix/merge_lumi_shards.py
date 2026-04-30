"""Merge experiment-J shard JSONs into matrix_results.json."""
import glob
import json
import os
import statistics
import collections


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    merged = {}
    for shard in sorted(glob.glob(os.path.join(HERE, "matrix_lumi_shard*.json"))):
        d = json.load(open(shard))
        merged.update(d)
        print(f"  {os.path.basename(shard)}: {len(d)} entries -> total {len(merged)}")
    out = os.path.join(HERE, "matrix_results.json")
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nWrote {out} with {len(merged)} entries.")

    # best-of-N per (label, cell)
    by_cell = collections.defaultdict(list)
    for v in merged.values():
        if "aep_gwh" in v and v.get("feasible"):
            key = (v["label"], v["farm"], v["n"], v["rose"])
            by_cell[key].append(v["aep_gwh"])

    best_of_n = {}
    for (label, farm, n, rose), aeps in by_cell.items():
        best_of_n[f"{label}|{farm}_n{n}_rose{rose}"] = {
            "best_of_n": max(aeps),
            "mean": round(statistics.mean(aeps), 3),
            "n_seeds": len(aeps),
        }
    out2 = os.path.join(HERE, "matrix_bestof.json")
    with open(out2, "w") as f:
        json.dump(best_of_n, f, indent=2)
    print(f"Wrote {out2} with {len(best_of_n)} (label, cell) entries.")


if __name__ == "__main__":
    main()
