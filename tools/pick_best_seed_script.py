"""Scan a set of agent output directories, return the path of the
single script with the highest feasible held-out (ROWP) AEP.

Used by the iterative seed-evolution driver to pick the best script
from one generation of seed runs and use it as the hot-start for
the next generation.

Usage:
    python tools/pick_best_seed_script.py \
        --dirs results_agent_r1d7_gen1_s*
    # Prints the best iter_NNN.py path to stdout, plus summary to stderr

Selection criterion (in order):
  1. Prefer the highest FEASIBLE rowp_aep
  2. If no feasible rowp, fall back to highest feasible train_aep
  3. If nothing feasible at all, fall back to highest train_aep
"""
import argparse
import glob
import json
import os
import sys


def scan_dir(d: str):
    """Return list of (iter_path, train_aep, train_feas, rowp_aep, rowp_feas)."""
    log = os.path.join(d, "attempt_log.json")
    if not os.path.exists(log):
        return []
    try:
        atts = json.load(open(log))
    except Exception:
        return []
    out = []
    for a in atts:
        if "error" in a and "train_aep" not in a:
            continue
        iter_path = os.path.join(d, f"iter_{a.get('attempt', 0):03d}.py")
        if not os.path.exists(iter_path):
            continue
        out.append({
            "path": iter_path,
            "dir": os.path.basename(d),
            "attempt": a.get("attempt"),
            "train_aep": a.get("train_aep"),
            "train_feas": bool(a.get("train_feasible")),
            "rowp_aep": a.get("rowp_aep"),
            "rowp_feas": bool(a.get("rowp_feasible")),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True,
                   help="glob patterns for agent output dirs")
    p.add_argument("--out-symlink", default=None,
                   help="if set, update this symlink to point at the best script")
    args = p.parse_args()

    expanded = []
    for pat in args.dirs:
        expanded.extend(sorted(glob.glob(pat)))
    if not expanded:
        print("No directories matched", file=sys.stderr)
        sys.exit(1)

    all_attempts = []
    for d in expanded:
        all_attempts.extend(scan_dir(d))

    print(f"Scanned {len(expanded)} dirs, found {len(all_attempts)} attempts",
          file=sys.stderr)
    if not all_attempts:
        print("No attempts in any dir", file=sys.stderr)
        sys.exit(2)

    # Tier 1: best feasible ROWP
    feas_rowp = [a for a in all_attempts
                 if a["rowp_feas"] and a["rowp_aep"] is not None]
    if feas_rowp:
        best = max(feas_rowp, key=lambda a: a["rowp_aep"])
        tier = "feasible_rowp"
    else:
        # Tier 2: best feasible train AEP
        feas_train = [a for a in all_attempts
                      if a["train_feas"] and a["train_aep"] is not None]
        if feas_train:
            best = max(feas_train, key=lambda a: a["train_aep"])
            tier = "feasible_train"
        else:
            # Tier 3: best train AEP regardless of feasibility
            any_train = [a for a in all_attempts
                         if a["train_aep"] is not None]
            if not any_train:
                print("No scored attempts found", file=sys.stderr)
                sys.exit(3)
            best = max(any_train, key=lambda a: a["train_aep"])
            tier = "any_train"

    print(f"Best by {tier}: {best['path']} "
          f"(dir={best['dir']}, attempt={best['attempt']}, "
          f"train={best['train_aep']}, rowp={best['rowp_aep']}, "
          f"train_feas={best['train_feas']}, rowp_feas={best['rowp_feas']})",
          file=sys.stderr)

    if args.out_symlink:
        if os.path.lexists(args.out_symlink):
            os.remove(args.out_symlink)
        os.symlink(os.path.abspath(best["path"]), args.out_symlink)
        print(f"Symlink updated: {args.out_symlink} -> {best['path']}",
              file=sys.stderr)

    # Print path to stdout for shell consumption
    print(best["path"])


if __name__ == "__main__":
    main()
