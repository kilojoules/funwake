"""Merge per-GCD random-search shards into a single attempt_log.json.

Each shard directory has its own iter_NNN.py files and attempt_log.json
with attempts numbered 1..N. The merged output renumbers attempts
sequentially and copies the iter files to avoid collisions.
"""
import argparse
import json
import os
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard-dirs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    merged = []
    next_id = 1

    for shard in args.shard_dirs:
        log_path = os.path.join(shard, "attempt_log.json")
        if not os.path.exists(log_path):
            print(f"  skip: {log_path} (missing)")
            continue
        with open(log_path) as f:
            shard_attempts = json.load(f)
        for entry in shard_attempts:
            old_id = entry.get("attempt")
            new_id = next_id
            new_entry = dict(entry)
            new_entry["attempt"] = new_id
            new_entry["shard"] = os.path.basename(shard)
            merged.append(new_entry)

            if old_id is not None:
                src = os.path.join(shard, f"iter_{old_id:03d}.py")
                dst = os.path.join(args.output_dir, f"iter_{new_id:03d}.py")
                if os.path.exists(src):
                    shutil.copy2(src, dst)

            next_id += 1
        print(f"  {shard}: +{len(shard_attempts)} (total {next_id-1})")

    out_log = os.path.join(args.output_dir, "attempt_log.json")
    with open(out_log, "w") as f:
        json.dump(merged, f, indent=2)

    scored = [a for a in merged if "train_aep" in a]
    rowp = [a for a in scored if "rowp_aep" in a]
    feasible_rowp = [a for a in rowp if a.get("rowp_feasible")]

    print()
    print(f"Merged {len(merged)} attempts into {out_log}")
    print(f"  scored (train): {len(scored)}")
    print(f"  scored (ROWP):  {len(rowp)}")
    print(f"  feasible ROWP:  {len(feasible_rowp)}")
    if scored:
        best_t = max(scored, key=lambda a: a["train_aep"])
        print(f"  best train:     {best_t['train_aep']:.1f} GWh")
    if feasible_rowp:
        best_r = max(feasible_rowp, key=lambda a: a["rowp_aep"])
        print(f"  best feas ROWP: {best_r['rowp_aep']:.1f} GWh")


if __name__ == "__main__":
    main()
