"""Merge LUMI shard JSONs from run_matrix_codex_lumi.sbatch.

Usage:
    pixi run python experiments/H_codex_six_runs/merge_lumi_shards.py
"""
import glob
import json
import os


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    merged = {}
    for shard in sorted(glob.glob(os.path.join(HERE, "matrix_lumi_shard*.json"))):
        d = json.load(open(shard))
        merged.update(d)
        print(f"  merged {os.path.basename(shard)}: {len(d)} entries (running total {len(merged)})")
    out = os.path.join(HERE, "matrix_lumi_results.json")
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {out} with {len(merged)} cells.")


if __name__ == "__main__":
    main()
