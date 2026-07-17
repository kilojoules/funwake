"""Evaluate full-optimizer optimize() scripts on every cell of the
farm × N × wind-rose matrix (2 polygons × 4 roses × 8 N = 64 cells).

Full-optimizer analogue of tools/eval_matrix_schedules.py: runs each
champion's optimize(sim, n_target, boundary, ...) on each cell and scores
the resulting layout against that cell's rose — WITHOUT --schedule-only.

The codex full-opt champions were written for the DEI N=50 training farm;
scoring them across the matrix tests whether the richer optimize()
interface generalizes the way the schedule-only dual-bump schedule does.

Parallel local runner (ProcessPool over cells). Resume-safe: one merged
JSON, skips cells already scored.

Usage:
    pixi run python tools/eval_matrix_fullopt.py --workers 3 --timeout 600
    pixi run python tools/eval_matrix_fullopt.py --max-n 80   # N<=80 subset
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (label, script_path) — the three codex full-optimizer champions.
CHAMPIONS = [
    ("codex_fullopt_run1 (iter_049)", "results_agent_codex_fullopt_run1/iter_049.py"),
    ("codex_fullopt_run2 (iter_056)", "results_agent_codex_fullopt_run2/iter_056.py"),
    ("codex_fullopt_run3 (iter_014)", "results_agent_codex_fullopt_run3/iter_014.py"),
]


def score_one(script_rel, problem_rel, timeout):
    env = {
        **os.environ,
        "JAX_ENABLE_X64": "True",
        "JAX_PLATFORMS": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONPATH": (
            os.path.join(PROJECT_ROOT, "playground", "pixwake", "src")
            + ":" + os.environ.get("PYTHONPATH", "")
        ),
    }
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "tools", "run_optimizer.py"),
        os.path.join(PROJECT_ROOT, script_rel),
        "--problem", os.path.join(PROJECT_ROOT, problem_rel),
        "--timeout", str(timeout),
        "--log", "/dev/null",
        # NOTE: no --schedule-only — full-optimizer mode.
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout + 90, env=env, cwd=PROJECT_ROOT)
        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            return {"error": (r.stderr or r.stdout or "no output")[:200]}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-n", type=int, default=None,
                    help="Skip cells with N > max-n (80 matches the schedule matrix figure).")
    ap.add_argument("--out", default="results/matrix/codex_fullopt_matrix.json")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(PROJECT_ROOT, "results", "matrix", "manifest.json")))
    cells = manifest["cells"]
    if args.max_n is not None:
        cells = [c for c in cells if c["n"] <= args.max_n]

    out_path = os.path.join(PROJECT_ROOT, args.out)
    results = {}
    if os.path.exists(out_path):
        results = json.load(open(out_path))

    # Build the task list, skipping already-scored (champion, cell) pairs.
    tasks = []
    for label, script in CHAMPIONS:
        for cell in cells:
            key = f"{label}|{cell['farm']}_n{cell['n']}_rose{cell['rose']}"
            if key in results and ("aep_gwh" in results[key] or "error" in results[key]):
                continue
            tasks.append((key, label, script, cell))

    total = len(CHAMPIONS) * len(cells)
    print(f"{len(tasks)} tasks to run ({total - len(tasks)} already done), "
          f"{args.workers} workers, timeout={args.timeout}s", flush=True)

    t_start = time.time()
    done = 0
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(score_one, script, cell["path"], args.timeout): (key, label, cell)
            for (key, label, script, cell) in tasks
        }
        for fut in cf.as_completed(futs):
            key, label, cell = futs[fut]
            r = fut.result()
            entry = {
                "label": label,
                "farm": cell["farm"], "n": cell["n"], "rose": cell["rose"],
            }
            entry.update(r)
            results[key] = entry
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            done += 1
            if "aep_gwh" in r:
                feas = "OK" if r.get("feasible") else "INFEAS"
                tag = f"AEP={r['aep_gwh']:.1f} {feas}"
            else:
                tag = f"ERR {r.get('error', '')[:50]}"
            print(f"[{done}/{len(tasks)}] {label} @ {cell['farm']} "
                  f"n={cell['n']} rose={cell['rose']}: {tag}", flush=True)

    print(f"\nWall: {(time.time()-t_start)/60:.1f} min. Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
