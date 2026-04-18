"""Terminal-alpha ablation: vary alpha scale while holding LR fixed.

Tests whether high terminal penalty weight is causally necessary for
held-out feasibility, or merely correlated with other schedule properties.

Takes Claude iter_192's schedule, multiplies alpha by a constant factor,
and scores on both DEI (training) and ROWP (held-out).

Usage:
    python tools/alpha_ablation.py --output-dir results_alpha_ablation
"""
import argparse
import json
import os
import sys
import time

FACTORS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

SCHEDULE_TEMPLATE = '''"""Alpha ablation: factor={factor}.

Claude iter_192 schedule with alpha multiplied by {factor}.
LR schedule, beta1, beta2 held fixed.
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps
    lr_init = 4.0 * lr0
    lr_min = lr_init / 10000.0

    # Warmup + cosine (UNCHANGED from iter_192)
    warmup_end = 0.05
    warmup_lr = lr_init * t / warmup_end
    cosine_t = (t - warmup_end) / (1.0 - warmup_end)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)

    bump1 = 0.2 * lr_init * jnp.exp(-0.5 * ((t - 0.5) / 0.04) ** 2)
    bump2 = 0.3 * lr_init * jnp.exp(-0.5 * ((t - 0.75) / 0.05) ** 2)
    lr = lr_base + bump1 + bump2

    # Alpha with dip — SCALED by factor {factor}
    alpha_base = 5.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    late = jnp.maximum(t - 0.5, 0.0) / 0.5
    alpha_extra = 3.0 * alpha0 * late ** 2

    dip = 0.5 * jnp.exp(-0.5 * ((t - 0.6) / 0.04) ** 2)
    alpha = {factor} * (alpha_base + alpha_extra) * (1.0 - dip)

    beta1 = 0.3
    beta2 = 0.5

    return lr, alpha, beta1, beta2
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="results_alpha_ablation")
    p.add_argument("--train-problem", default="playground/problem.json")
    p.add_argument("--rowp-problem", default="results/problem_rowp.json")
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    project_root = os.path.join(os.path.dirname(__file__), "..")
    tools_dir = os.path.dirname(__file__)
    pixwake_src = os.path.join(project_root, "playground", "pixwake", "src")

    results = []

    for factor in FACTORS:
        script_path = os.path.join(args.output_dir, f"alpha_{factor:.2f}.py")
        with open(script_path, "w") as f:
            f.write(SCHEDULE_TEMPLATE.format(factor=factor))

        entry = {"factor": factor}

        for label, problem in [("train", args.train_problem), ("rowp", args.rowp_problem)]:
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": f"{pixwake_src}:{os.environ.get('PYTHONPATH', '')}",
                "JAX_ENABLE_X64": "True",
                "HOME": os.environ.get("HOME", ""),
                "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
            }
            import subprocess
            try:
                t0 = time.time()
                result = subprocess.run(
                    [sys.executable, os.path.join(tools_dir, "run_optimizer.py"),
                     os.path.abspath(script_path),
                     "--problem", os.path.abspath(problem),
                     "--timeout", str(args.timeout),
                     "--log", "/dev/null",
                     "--schedule-only"],
                    capture_output=True, text=True,
                    timeout=args.timeout + 30, env=env, cwd=project_root)
                data = json.loads(result.stdout)
                entry[f"{label}_aep"] = data.get("aep_gwh")
                entry[f"{label}_feasible"] = data.get("feasible")
                entry[f"{label}_time"] = round(time.time() - t0, 1)
            except Exception as e:
                entry[f"{label}_error"] = str(e)[:200]

        results.append(entry)
        f_str = "FEAS" if entry.get("rowp_feasible") else "INFEAS"
        print(f"alpha×{factor:6.2f}  train={entry.get('train_aep', '?'):8.1f}  "
              f"rowp={entry.get('rowp_aep', '?'):8.1f}  rowp_{f_str}")

        # Save incrementally
        with open(os.path.join(args.output_dir, "ablation_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"{'Factor':>8}  {'Train AEP':>10}  {'ROWP AEP':>10}  {'ROWP Feas':>10}")
    for r in results:
        print(f"{r['factor']:8.2f}  {r.get('train_aep', 0):10.1f}  "
              f"{r.get('rowp_aep', 0):10.1f}  {str(r.get('rowp_feasible', '?')):>10}")


if __name__ == "__main__":
    main()
