"""Multi-seed validation of the top adversarial-selection candidates.

For each candidate script, scores it on (training cell, uniform stress
cell) for N initialization seeds. Reports mean/std/min/max of
train_aep, stress_aep, and min_gap, plus the % of seeds with both
cells feasible.

Cheap: ~30-45s per (script, seed) ≈ 25 min for 4 scripts × 5 seeds.

Usage:
    pixi run python tools/validate_adversarial_top.py \\
        --scripts results_agent_schedule_adversarial_local/iter_008.py \\
                  results_agent_schedule_adversarial_local/iter_011.py \\
                  results_agent_schedule_adversarial_local/iter_013.py \\
                  results_agent_schedule_adversarial_local/iter_014.py \\
        --n-seeds 5 \\
        --output results/adversarial_validation.json
"""
import argparse
import json
import os
import statistics
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def score_one(script, seed, train_problem, stress_problem,
              stress_baseline, timeout):
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "tools", "run_optimizer.py"),
        os.path.abspath(script),
        "--schedule-only",
        "--problem", os.path.abspath(train_problem),
        "--stress-problem", os.path.abspath(stress_problem),
        "--stress-baseline", str(stress_baseline),
        "--timeout", str(timeout),
        "--seed", str(seed),
        "--log", "/dev/null",
    ]
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout * 2 + 60, env=env,
                           cwd=PROJECT_ROOT)
        return json.loads(r.stdout)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)[:200]}


def summarize(rows):
    """Compute robust summary stats over per-seed rows."""
    def stats(vals):
        if not vals:
            return None
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            return None
        return {
            "n": len(vals),
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    train_aeps = [r.get("aep_gwh") for r in rows if "aep_gwh" in r]
    stress_aeps = [r.get("stress_aep_gwh") for r in rows if "stress_aep_gwh" in r]
    min_gaps = [r.get("min_gap") for r in rows
                 if r.get("min_gap") is not None]
    train_feas = sum(1 for r in rows if r.get("feasible"))
    stress_feas = sum(1 for r in rows if r.get("stress_feasible"))
    both_feas = sum(1 for r in rows
                     if r.get("feasible") and r.get("stress_feasible"))
    return {
        "n_seeds": len(rows),
        "train_aep": stats(train_aeps),
        "stress_aep": stats(stress_aeps),
        "min_gap": stats(min_gaps),
        "train_feasible_rate": train_feas / len(rows) if rows else 0,
        "stress_feasible_rate": stress_feas / len(rows) if rows else 0,
        "both_feasible_rate": both_feas / len(rows) if rows else 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scripts", nargs="+", required=True)
    p.add_argument("--train-problem", default="playground/problem.json")
    p.add_argument("--stress-problem",
                   default="results/matrix/problem_dei_n50_roseuniform.json")
    p.add_argument("--stress-baseline", type=float, default=5599.79)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    results = {}
    if os.path.exists(args.output):
        results = json.load(open(args.output))
        print(f"Resuming from {args.output} ({len(results)} scripts already done)")

    for script in args.scripts:
        label = os.path.basename(script).replace(".py", "")
        if label in results and "summary" in results[label]:
            print(f"[skip] {label} already done")
            continue
        print(f"\n=== {label} ===")
        rows = []
        for seed in range(args.n_seeds):
            print(f"  seed={seed} ... ", end="", flush=True)
            r = score_one(script, seed, args.train_problem,
                          args.stress_problem, args.stress_baseline,
                          args.timeout)
            if "error" in r:
                print(f"ERROR ({r['error'][:80]})")
            else:
                print(f"train={r.get('aep_gwh'):.2f} "
                      f"({'feas' if r.get('feasible') else 'INFEAS'}) "
                      f"stress={r.get('stress_aep_gwh'):.2f} "
                      f"({'feas' if r.get('stress_feasible') else 'INFEAS'}) "
                      f"min_gap={r.get('min_gap')}")
            rows.append(r)

        results[label] = {
            "script": script,
            "seeds": list(range(args.n_seeds)),
            "rows": rows,
            "summary": summarize(rows),
        }
        # Persist incrementally
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"{'script':<25} {'both_feas':>10} {'train_aep':>20} "
          f"{'stress_aep':>20} {'min_gap':>20}")
    for label, data in results.items():
        s = data.get("summary", {})
        ta = s.get("train_aep") or {}
        sa = s.get("stress_aep") or {}
        mg = s.get("min_gap") or {}
        print(f"{label:<25} {s.get('both_feasible_rate', 0):>10.0%} "
              f"{ta.get('mean', 0):>10.2f}±{ta.get('std', 0):.2f} "
              f"{sa.get('mean', 0):>10.2f}±{sa.get('std', 0):.2f} "
              f"{mg.get('mean', 0):>+10.2f}±{mg.get('std', 0):.2f}")


if __name__ == "__main__":
    main()
