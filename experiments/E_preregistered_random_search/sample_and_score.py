"""Sample 320 schedules from the pre-registered Fourier family and score
each on DEI (training) + ROWP (held-out).

Resume-safe: existing sampled scripts are NOT regenerated; existing
scored entries are skipped.

Usage:
    pixi run python experiments/E_preregistered_random_search/sample_and_score.py \
        --n-samples 320 --sampler-seed 0
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

# Local module
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from family import render, K, N_TOTAL  # noqa: E402


PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
TRAIN_PROBLEM = os.path.join(PROJECT_ROOT, "playground", "problem.json")
ROWP_PROBLEM  = os.path.join(PROJECT_ROOT, "results", "problem_rowp.json")


def sample_coeffs(rng):
    """Draw N_TOTAL coefficients per the pre-registered prior."""
    coeffs = np.zeros(N_TOTAL)
    n_per = 1 + 2 * K
    for o in range(4):  # outputs
        off = o * n_per
        coeffs[off] = rng.normal(0, 0.5)        # c0
        for k in range(1, K + 1):
            coeffs[off + k]         = rng.normal(0, 0.5 / k)   # a_k
            coeffs[off + K + k]     = rng.normal(0, 0.5 / k)   # b_k
    return coeffs


def run_optimizer(script_path, problem_path, timeout, init_seed=0):
    pixwake_src = os.path.join(PROJECT_ROOT, "playground", "pixwake", "src")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": f"{pixwake_src}:{os.environ.get('PYTHONPATH','')}",
        "JAX_ENABLE_X64": "True",
        "HOME": os.environ.get("HOME", ""),
    }
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "tools", "run_optimizer.py"),
           script_path, "--problem", problem_path, "--timeout", str(timeout),
           "--seed", str(init_seed), "--log", "/dev/null", "--schedule-only"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout + 30, env=env, cwd=PROJECT_ROOT)
        return json.loads(r.stdout)
    except Exception as e:
        return {"error": str(e)[:200]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=320)
    p.add_argument("--sampler-seed", type=int, default=0)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--scripts-dir", default=os.path.join(HERE, "scripts"))
    p.add_argument("--results", default=os.path.join(HERE, "results.json"))
    p.add_argument("--timeout", type=int, default=180)
    args = p.parse_args()

    os.makedirs(args.scripts_dir, exist_ok=True)

    rng = np.random.default_rng(args.sampler_seed)
    existing = json.load(open(args.results)) if os.path.exists(args.results) else {}

    for i in range(args.n_samples):
        sid = f"sample_{i:04d}"
        path = os.path.join(args.scripts_dir, f"{sid}.py")

        # generate or load coeffs
        if not os.path.exists(path):
            coeffs = sample_coeffs(rng)
            src = render(i, coeffs.tolist())
            with open(path, "w") as f:
                f.write(src)
        else:
            # Re-derive coeffs deterministically by reseeding through i samples.
            # Simpler: ensure rng is advanced regardless. We just consume noise.
            sample_coeffs(rng)

        # score on train if not present
        rec = existing.get(sid, {})
        if "train_aep" not in rec:
            t0 = time.time()
            r = run_optimizer(path, TRAIN_PROBLEM, args.timeout, args.init_seed)
            rec["train_aep"]      = r.get("aep_gwh")
            rec["train_feasible"] = r.get("feasible")
            rec["train_time_s"]   = round(time.time() - t0, 1)
            if "error" in r:
                rec["train_error"] = r["error"]
        # score on rowp only if train was feasible (skip clearly bad samples)
        if rec.get("train_feasible") and "rowp_aep" not in rec:
            t0 = time.time()
            r = run_optimizer(path, ROWP_PROBLEM, args.timeout, args.init_seed)
            rec["rowp_aep"]      = r.get("aep_gwh")
            rec["rowp_feasible"] = r.get("feasible")
            rec["rowp_time_s"]   = round(time.time() - t0, 1)
            if "error" in r:
                rec["rowp_error"] = r["error"]

        existing[sid] = rec

        if (i + 1) % 10 == 0 or i + 1 == args.n_samples:
            with open(args.results, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"[E] {i+1}/{args.n_samples}  train={rec.get('train_aep','?')}  rowp={rec.get('rowp_aep','?')}")

    with open(args.results, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[E] Wrote {args.results}")


if __name__ == "__main__":
    main()
