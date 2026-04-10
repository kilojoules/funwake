#!/usr/bin/env python
"""Post-process: score all iter_*.py scripts on the held-out ROWP farm.

Run AFTER the agent loop completes. This backfills rowp_aep, rowp_feasible,
and rowp_time into the attempt_log.json for each results directory.

The agent never sees ROWP scores — this runs after the fact.

Usage:
    python tools/backfill_rowp.py results_agent_qwen2_5-coder-32b_s1/
    python tools/backfill_rowp.py results_agent_*_s*/   # all at once
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description="Backfill ROWP scores into attempt logs")
    p.add_argument("dirs", nargs="+", help="Result directories to process")
    p.add_argument("--problem", default="results/problem_rowp.json",
                   help="Path to ROWP problem JSON")
    p.add_argument("--timeout", type=int, default=180,
                   help="Timeout per script (default: 180s)")
    p.add_argument("--schedule-only", action="store_true",
                   help="Scripts contain schedule_fn, not optimize")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be scored without running")
    args = p.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), "..")
    tools_dir = os.path.dirname(__file__)
    pixwake_src = os.path.join(project_root, "playground", "pixwake", "src")

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": f"{pixwake_src}:{os.environ.get('PYTHONPATH', '')}",
        "JAX_ENABLE_X64": "True",
        "HOME": os.environ.get("HOME", ""),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }

    for result_dir in args.dirs:
        result_dir = result_dir.rstrip("/")
        log_path = os.path.join(result_dir, "attempt_log.json")

        if not os.path.exists(log_path):
            print(f"Skipping {result_dir}: no attempt_log.json")
            continue

        with open(log_path) as f:
            attempts = json.load(f)

        scripts = sorted(glob.glob(os.path.join(result_dir, "iter_*.py")))
        print(f"\n{result_dir}: {len(attempts)} attempts, {len(scripts)} scripts")

        # Build a map of attempt number -> log entry
        attempt_map = {a["attempt"]: a for a in attempts}

        updated = 0
        for script in scripts:
            # Extract attempt number from filename (iter_001.py -> 1)
            basename = os.path.basename(script)
            try:
                attempt_num = int(basename.replace("iter_", "").replace(".py", ""))
            except ValueError:
                continue

            entry = attempt_map.get(attempt_num)
            if entry is None:
                continue

            # Skip if already scored or errored
            if "rowp_aep" in entry:
                print(f"  iter_{attempt_num:03d}: already scored (rowp={entry['rowp_aep']:.1f})")
                continue
            if "error" in entry:
                print(f"  iter_{attempt_num:03d}: skipped (error during training)")
                continue

            if args.dry_run:
                print(f"  iter_{attempt_num:03d}: would score")
                continue

            # Score on ROWP
            print(f"  iter_{attempt_num:03d}: scoring on ROWP...", end="", flush=True)
            t0 = time.time()
            try:
                cmd = [
                    sys.executable,
                    os.path.join(tools_dir, "run_optimizer.py"),
                    os.path.abspath(script),
                    "--problem", os.path.abspath(args.problem),
                    "--timeout", str(args.timeout),
                    "--log", "/dev/null",  # don't append to attempt_log
                ]
                if args.schedule_only:
                    cmd.append("--schedule-only")

                result = subprocess.run(
                    cmd,
                    capture_output=True, text=True,
                    timeout=args.timeout + 30,
                    env=env, cwd=project_root,
                )
                elapsed = time.time() - t0
                data = json.loads(result.stdout)

                if "aep_gwh" in data:
                    entry["rowp_aep"] = data["aep_gwh"]
                    entry["rowp_feasible"] = data.get("feasible")
                    entry["rowp_time"] = data.get("time_s")
                    print(f" {data['aep_gwh']:.1f} GWh, "
                          f"feasible={data.get('feasible')}, {elapsed:.0f}s")
                    updated += 1
                else:
                    entry["rowp_error"] = data.get("error", "unknown")[:200]
                    print(f" error: {data.get('error', '')[:60]}")

            except subprocess.TimeoutExpired:
                entry["rowp_error"] = f"Timeout after {args.timeout}s"
                print(f" timeout ({args.timeout}s)")
            except Exception as e:
                entry["rowp_error"] = str(e)[:200]
                print(f" exception: {e}")

        # Write updated log
        if updated > 0 and not args.dry_run:
            with open(log_path, "w") as f:
                json.dump(attempts, f, indent=2)
            print(f"  Updated {updated} entries in {log_path}")


if __name__ == "__main__":
    main()
