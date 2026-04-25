"""Data cleansing pipeline for FunWake attempt logs.

1. Separates N=50 training entries from generalization evaluations
2. Backfills missing ROWP scores for scripts that exist on disk
3. Writes cleaned logs

Usage:
    pixi run python tools/cleanse_and_backfill.py
    pixi run python tools/cleanse_and_backfill.py --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import time


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWP_PROBLEM = os.path.join(PROJECT_ROOT, "results", "problem_rowp.json")


def score_on_rowp(script_path, timeout=180):
    """Score a schedule script on the ROWP held-out farm."""
    tools_dir = os.path.join(PROJECT_ROOT, "tools")
    pixwake_src = os.path.join(PROJECT_ROOT, "playground", "pixwake", "src")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": f"{pixwake_src}:{os.environ.get('PYTHONPATH', '')}",
        "JAX_ENABLE_X64": "True",
        "HOME": os.environ.get("HOME", ""),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(tools_dir, "run_optimizer.py"),
             os.path.abspath(script_path),
             "--problem", os.path.abspath(ROWP_PROBLEM),
             "--timeout", str(timeout),
             "--log", "/dev/null",
             "--schedule-only"],
            capture_output=True, text=True,
            timeout=timeout + 30, env=env, cwd=PROJECT_ROOT)
        return json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e)[:200]}


def cleanse_log(log_path, results_dir, dry_run=False):
    """Cleanse one attempt log: separate N=50 from generalization, backfill ROWP."""
    log = json.load(open(log_path))
    name = os.path.basename(results_dir)

    scored = [e for e in log if "train_aep" in e]
    n50 = [e for e in scored if e["train_aep"] < 5700]
    gen = [e for e in scored if e["train_aep"] >= 5700]
    errors = [e for e in log if "error" in e and "train_aep" not in e]

    print(f"\n=== {name} ===")
    print(f"  Total: {len(log)}, N=50: {len(n50)}, generalization: {len(gen)}, errors: {len(errors)}")

    # Find scripts that need ROWP backfill
    scripts_on_disk = set(os.listdir(results_dir))
    to_backfill = []
    for e in n50:
        if e.get("rowp_aep") is not None:
            continue
        if not e.get("train_feasible"):
            continue
        fname = f"iter_{e['attempt']:03d}.py"
        fpath = os.path.join(results_dir, fname)
        if fname in scripts_on_disk:
            to_backfill.append((e, fpath))

    print(f"  With ROWP: {sum(1 for e in n50 if e.get('rowp_aep') is not None)}")
    print(f"  Missing ROWP (feasible, script exists): {len(to_backfill)}")

    if dry_run:
        print("  [DRY RUN] Would backfill:")
        for e, fpath in to_backfill[:10]:
            print(f"    attempt {e['attempt']}: train={e['train_aep']:.1f} -> {os.path.basename(fpath)}")
        if len(to_backfill) > 10:
            print(f"    ... and {len(to_backfill) - 10} more")
        return

    # Backfill
    filled = 0
    for e, fpath in to_backfill:
        print(f"  Backfilling attempt {e['attempt']} ({os.path.basename(fpath)})...", end=" ", flush=True)
        t0 = time.time()
        result = score_on_rowp(fpath)
        elapsed = time.time() - t0

        if "error" in result:
            print(f"ERROR ({elapsed:.0f}s): {result['error'][:60]}")
            continue

        e["rowp_aep"] = result.get("aep_gwh")
        e["rowp_feasible"] = result.get("feasible")
        e["rowp_time"] = round(elapsed, 1)
        e["rowp_backfilled"] = True
        filled += 1
        feas = "FEAS" if result.get("feasible") else "INFEAS"
        print(f"ROWP={result.get('aep_gwh', 0):.1f} {feas} ({elapsed:.0f}s)")

    print(f"  Backfilled: {filled}/{len(to_backfill)}")

    # Write cleaned logs
    clean_path = os.path.join(results_dir, "attempt_log_clean.json")
    gen_path = os.path.join(results_dir, "attempt_log_generalization.json")

    with open(clean_path, "w") as f:
        json.dump(n50 + errors, f, indent=2)
    print(f"  Wrote {clean_path} ({len(n50) + len(errors)} entries)")

    if gen:
        with open(gen_path, "w") as f:
            json.dump(gen, f, indent=2)
        print(f"  Wrote {gen_path} ({len(gen)} entries)")

    # Also update the original log with backfilled data
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Updated {log_path} with backfill data")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be done without running")
    args = p.parse_args()

    dirs = [
        "results_agent_schedule_only_5hr",
        "results_agent_gemini_cli_5hr",
    ]

    for d in dirs:
        results_dir = os.path.join(PROJECT_ROOT, d)
        log_path = os.path.join(results_dir, "attempt_log.json")
        if not os.path.exists(log_path):
            print(f"Skipping {d}: no attempt_log.json")
            continue
        cleanse_log(log_path, results_dir, dry_run=args.dry_run)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
