"""Optimize parameters within the "dual Gaussian bump" schedule family.

Takes Claude's iter_192 structure as a parameterized family and searches
for the best parameter values using scipy's differential_evolution. The
optimizer only sees training AEP — held-out (ROWP) is evaluated post-hoc,
matching the LLM's rules.

Parameters (14 total):
  K        : initial LR multiplier [1, 8]
  logM     : log10(lr_init / lr_min) [2, 5]
  W        : warmup fraction [0, 0.15]
  A1, A2   : bump amplitudes [0, 0.5]
  c1, c2   : bump centers [0.1, 0.95]
  w1, w2   : bump widths [0.01, 0.15]
  C        : alpha coupling strength [1, 10]
  D        : alpha quadratic boost [0, 20]
  B1       : beta1 [0, 0.95]
  B2       : beta2 [0.5, 0.9995]

Usage:
    python tools/optimize_bump_family.py --max-iter 20 --popsize 15 \
        --output-dir results_bump_opt
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution


SCHEDULE_CODE = '''"""Bump-family schedule, optimized parameters."""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    K = {K}
    logM = {logM}
    W = {W}
    A1 = {A1}
    A2 = {A2}
    c1 = {c1}
    c2 = {c2}
    w1 = {w1}
    w2 = {w2}
    C = {C}
    D = {D}
    B1 = {B1}
    B2 = {B2}

    lr_init = K * lr0
    lr_min = lr_init / (10.0 ** logM)

    warmup_lr = lr_init * t / jnp.maximum(W, 1e-6)
    cosine_t = (t - W) / jnp.maximum(1.0 - W, 1e-6)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < W, warmup_lr, cosine_lr)

    bump1 = A1 * lr_init * jnp.exp(-0.5 * ((t - c1) / w1) ** 2)
    bump2 = A2 * lr_init * jnp.exp(-0.5 * ((t - c2) / w2) ** 2)
    lr = lr_base + bump1 + bump2
    lr = jnp.maximum(lr, 1e-10)

    alpha_base = C * alpha0 * lr_init / lr
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = D * alpha0 * late ** 2
    alpha = alpha_base + alpha_extra

    return lr, alpha, B1, B2
'''


PARAM_NAMES = ["K", "logM", "W", "A1", "A2", "c1", "c2", "w1", "w2", "C", "D", "B1", "B2"]
BOUNDS = [
    (1.0, 8.0),    # K
    (2.0, 5.0),    # logM
    (0.0, 0.15),   # W
    (0.0, 0.5),    # A1
    (0.0, 0.5),    # A2
    (0.1, 0.95),   # c1
    (0.1, 0.95),   # c2
    (0.01, 0.15),  # w1
    (0.01, 0.15),  # w2
    (1.0, 10.0),   # C
    (0.0, 20.0),   # D
    (0.0, 0.95),   # B1
    (0.5, 0.9995), # B2
]


def params_to_dict(x):
    return {name: float(v) for name, v in zip(PARAM_NAMES, x)}


def render_schedule(params):
    return SCHEDULE_CODE.format(**{k: f"{v:.6f}" for k, v in params.items()})


def score_on_farm(script_path, problem_path, timeout=120):
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
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(tools_dir, "run_optimizer.py"),
             os.path.abspath(script_path),
             "--problem", os.path.abspath(problem_path),
             "--timeout", str(timeout),
             "--log", "/dev/null",
             "--schedule-only"],
            capture_output=True, text=True, timeout=timeout + 30,
            env=env, cwd=project_root)
        return json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e)[:200]}


class Optimizer:
    def __init__(self, output_dir, train_problem, rowp_problem, timeout):
        self.output_dir = output_dir
        self.train_problem = train_problem
        self.rowp_problem = rowp_problem
        self.timeout = timeout
        self.history = []
        self.best_train = -np.inf
        self.best_x = None
        self.eval_count = 0
        os.makedirs(output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, "bump_opt_log.json")

    def objective(self, x):
        self.eval_count += 1
        params = params_to_dict(x)
        script_path = os.path.join(self.output_dir, f"iter_{self.eval_count:03d}.py")
        with open(script_path, "w") as f:
            f.write(render_schedule(params))

        t0 = time.time()
        result = score_on_farm(script_path, self.train_problem, self.timeout)
        elapsed = time.time() - t0

        entry = {
            "eval": self.eval_count,
            "params": params,
            "timestamp": time.time(),
        }

        if "error" in result:
            entry["error"] = result["error"][:200]
            print(f"[{self.eval_count}] ERROR: {result['error'][:60]}")
            self.history.append(entry)
            self._save()
            return 1e9

        aep = result.get("aep_gwh", 0)
        feasible = result.get("feasible", False)
        entry["train_aep"] = aep
        entry["train_feasible"] = feasible
        entry["train_time"] = round(elapsed, 1)

        # Penalty for infeasibility
        if not feasible:
            aep -= 100.0

        if aep > self.best_train:
            self.best_train = aep
            self.best_x = x.copy()

        print(f"[{self.eval_count}] AEP={result.get('aep_gwh', 0):.1f} "
              f"feas={feasible} best={self.best_train:.1f} ({elapsed:.0f}s)")

        self.history.append(entry)
        self._save()
        return -aep

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump({
                "best_train": self.best_train,
                "best_x": self.best_x.tolist() if self.best_x is not None else None,
                "history": self.history,
            }, f, indent=2)

    def evaluate_rowp(self):
        """Score all history entries on ROWP (post-hoc)."""
        print("\n=== ROWP backfill ===")
        for entry in self.history:
            if "train_aep" not in entry or "rowp_aep" in entry:
                continue
            script_path = os.path.join(self.output_dir, f"iter_{entry['eval']:03d}.py")
            if not os.path.exists(script_path):
                continue
            result = score_on_farm(script_path, self.rowp_problem, self.timeout)
            if "aep_gwh" in result:
                entry["rowp_aep"] = result["aep_gwh"]
                entry["rowp_feasible"] = result.get("feasible")
                print(f"  eval {entry['eval']}: rowp={result['aep_gwh']:.1f} "
                      f"feas={result.get('feasible')}")
        self._save()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="results_bump_opt")
    p.add_argument("--train-problem", default="playground/problem.json")
    p.add_argument("--rowp-problem", default="results/problem_rowp.json")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max-iter", type=int, default=15)
    p.add_argument("--popsize", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-rowp", action="store_true")
    args = p.parse_args()

    opt = Optimizer(args.output_dir, args.train_problem, args.rowp_problem, args.timeout)

    # Claude iter_192 as x0 — good starting point inside the family
    claude_init = np.array([
        4.0,    # K
        4.0,    # logM (lr_min = lr_init / 10000)
        0.05,   # W (warmup)
        0.2,    # A1 (first bump amp)
        0.3,    # A2 (second bump amp)
        0.5,    # c1
        0.75,   # c2
        0.04,   # w1
        0.05,   # w2
        5.0,    # C (alpha coupling)
        3.0,    # D (quadratic boost)
        0.3,    # B1
        0.5,    # B2
    ])

    print("Starting differential evolution: "
          f"popsize={args.popsize}, maxiter={args.max_iter}, "
          f"max_evals={args.popsize * args.max_iter * len(PARAM_NAMES)}")
    print(f"Initial (Claude iter_192-inspired): {params_to_dict(claude_init)}")

    # First evaluate the init point
    opt.objective(claude_init)

    # Build initial population: claude_init + random perturbations within bounds
    rng = np.random.default_rng(args.seed)
    lower = np.array([b[0] for b in BOUNDS])
    upper = np.array([b[1] for b in BOUNDS])
    init_pop = np.zeros((args.popsize, len(PARAM_NAMES)))
    init_pop[0] = claude_init
    for i in range(1, args.popsize):
        init_pop[i] = rng.uniform(lower, upper)

    result = differential_evolution(
        opt.objective,
        bounds=BOUNDS,
        maxiter=args.max_iter,
        popsize=args.popsize,
        init=init_pop,
        seed=args.seed,
        tol=0.0,  # don't early-stop
        polish=False,  # skip L-BFGS polish (too expensive)
        workers=1,  # sequential (each eval is already slow)
        disp=True,
    )

    print("\n=== OPTIMIZATION DONE ===")
    print(f"Best train AEP: {opt.best_train:.1f}")
    print(f"Best params: {params_to_dict(opt.best_x)}")
    print(f"Total evaluations: {opt.eval_count}")

    if not args.skip_rowp:
        opt.evaluate_rowp()

    # Final summary
    scored = [e for e in opt.history if "train_aep" in e]
    rowp = [e for e in scored if "rowp_aep" in e]
    if scored:
        best_t = max(scored, key=lambda e: e["train_aep"])
        print(f"\nBest train in history: {best_t['train_aep']:.1f}")
        if rowp:
            best_r = max(rowp, key=lambda e: e["rowp_aep"])
            print(f"Best ROWP in history:  {best_r['rowp_aep']:.1f}")


if __name__ == "__main__":
    main()
