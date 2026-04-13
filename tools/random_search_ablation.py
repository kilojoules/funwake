"""Random search ablation: sample schedules from a parameterized family.

Family covers cosine/exponential/linear LR decay with optional warmup,
optional sinusoidal shake or Gaussian bumps, monotonic penalty ramps
(linear/quadratic/coupled), and random Adam betas.

Each sampled schedule is evaluated the same way as an LLM-generated one:
run the skeleton, score on the training farm and (optionally) ROWP.

Usage:
    python tools/random_search_ablation.py --n-samples 320 \
        --output-dir results_random_search_320
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time


# Schedule templates — each renders to a concrete schedule_fn source string
SCHEDULE_TEMPLATE = '''"""Random search sample {sample_id}.

Family params: {params}
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / total_steps

    # Learning rate base curve
    lr_init = {lr_mult} * lr0
    lr_final_ratio = {lr_final_ratio}
    lr_min = lr_init * lr_final_ratio

    warmup_frac = {warmup_frac}
    warmup_lr = lr_init * jnp.minimum(1.0, t / jnp.maximum(warmup_frac, 1e-6))

    post_warmup_t = jnp.maximum(t - warmup_frac, 0.0) / jnp.maximum(1.0 - warmup_frac, 1e-6)

    {decay_expression}

    lr_base = jnp.where(t < warmup_frac, warmup_lr, lr_decay)

    # Perturbation
    {perturbation_expression}

    lr = lr_base * (1.0 + perturbation)
    lr = jnp.maximum(lr, 1e-10)

    # Alpha: penalty weight
    {alpha_expression}

    beta1 = {beta1}
    beta2 = {beta2}

    return lr, alpha, beta1, beta2
'''


def sample_schedule(sample_id, rng):
    """Sample a random schedule from the parameterized family."""
    lr_mult = rng.uniform(0.5, 5.0)
    lr_final_ratio = 10 ** rng.uniform(-4, -1)  # 1e-4 to 1e-1
    warmup_frac = rng.choice([0.0, 0.02, 0.05, 0.10])

    # Decay type
    decay_type = rng.choice(["cosine", "exponential", "linear", "polynomial"])
    if decay_type == "cosine":
        decay_expr = "lr_decay = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * post_warmup_t))"
    elif decay_type == "exponential":
        decay_rate = rng.uniform(3, 10)
        decay_expr = f"lr_decay = lr_init * jnp.exp(-{decay_rate:.2f} * post_warmup_t) + lr_min"
    elif decay_type == "linear":
        decay_expr = "lr_decay = lr_init * (1.0 - post_warmup_t) + lr_min * post_warmup_t"
    else:  # polynomial
        power = rng.uniform(1.5, 4.0)
        decay_expr = f"lr_decay = lr_min + (lr_init - lr_min) * (1.0 - post_warmup_t) ** {power:.2f}"

    # Perturbation
    pert_type = rng.choice(["none", "sinusoidal", "gaussian_bumps"])
    if pert_type == "none":
        pert_expr = "perturbation = 0.0"
    elif pert_type == "sinusoidal":
        freq = rng.choice([10, 20, 40, 80, 160])
        amp = rng.uniform(0.05, 0.25)
        decay_form = rng.choice(["sqrt", "linear", "none"])
        if decay_form == "sqrt":
            pert_expr = f"perturbation = {amp:.3f} * jnp.sqrt(jnp.maximum(1.0 - t, 0.0)) * jnp.sin({freq} * jnp.pi * t)"
        elif decay_form == "linear":
            pert_expr = f"perturbation = {amp:.3f} * (1.0 - t) * jnp.sin({freq} * jnp.pi * t)"
        else:
            pert_expr = f"perturbation = {amp:.3f} * jnp.sin({freq} * jnp.pi * t)"
    else:  # gaussian_bumps
        n_bumps = rng.choice([1, 2])
        bump_exprs = []
        for i in range(n_bumps):
            center = rng.uniform(0.2, 0.85)
            width = rng.uniform(0.02, 0.08)
            amp = rng.uniform(0.1, 0.5)
            bump_exprs.append(f"{amp:.3f} * jnp.exp(-0.5 * ((t - {center:.3f}) / {width:.3f})**2)")
        pert_expr = "perturbation = " + " + ".join(bump_exprs)

    # Alpha ramp
    alpha_type = rng.choice(["coupled", "quadratic", "linear", "coupled_plus_ramp"])
    alpha_final_mult = 10 ** rng.uniform(1, 4)  # 10x to 10000x
    if alpha_type == "coupled":
        # alpha ~ alpha0 * lr0 / lr (TopFarm style)
        coupling = rng.uniform(1.0, 10.0)
        alpha_expr = f"alpha = {coupling:.2f} * alpha0 * lr0 / lr"
    elif alpha_type == "quadratic":
        alpha_expr = f"alpha = alpha0 * (1.0 + {alpha_final_mult:.1f} * t * t)"
    elif alpha_type == "linear":
        alpha_expr = f"alpha = alpha0 * (1.0 + {alpha_final_mult:.1f} * t)"
    else:  # coupled_plus_ramp
        coupling = rng.uniform(1.0, 5.0)
        alpha_expr = f"alpha = ({coupling:.2f} * alpha0 * lr0 / lr) + alpha0 * {alpha_final_mult:.1f} * t * t"

    # Adam betas
    beta1 = round(rng.uniform(0.0, 0.95), 3)
    beta2 = round(rng.uniform(0.5, 0.9999), 4)

    params = {
        "lr_mult": round(lr_mult, 3),
        "lr_final_ratio": f"{lr_final_ratio:.4g}",
        "warmup_frac": warmup_frac,
        "decay": decay_type,
        "perturbation": pert_type,
        "alpha": alpha_type,
        "alpha_final_mult": round(alpha_final_mult, 1),
        "beta1": beta1,
        "beta2": beta2,
    }

    code = SCHEDULE_TEMPLATE.format(
        sample_id=sample_id,
        params=json.dumps(params),
        lr_mult=f"{lr_mult:.4f}",
        lr_final_ratio=f"{lr_final_ratio:.6g}",
        warmup_frac=warmup_frac,
        decay_expression=decay_expr,
        perturbation_expression=pert_expr,
        alpha_expression=alpha_expr,
        beta1=beta1,
        beta2=beta2,
    )
    return code, params


def score_on_farm(script_path, problem_path, timeout=120, schedule_only=True):
    """Score a schedule script on a farm, return (aep, feasible, time) or error."""
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

    cmd = [
        sys.executable,
        os.path.join(tools_dir, "run_optimizer.py"),
        os.path.abspath(script_path),
        "--problem", os.path.abspath(problem_path),
        "--timeout", str(timeout),
        "--log", "/dev/null",
    ]
    if schedule_only:
        cmd.append("--schedule-only")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout + 30, env=env, cwd=project_root,
        )
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {"error": "Timeout"}
    except Exception as e:
        return {"error": str(e)[:200]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=320)
    p.add_argument("--output-dir", default="results_random_search")
    p.add_argument("--train-problem", default="playground/problem.json")
    p.add_argument("--rowp-problem", default="results/problem_rowp.json")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-rowp", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "attempt_log.json")

    rng = random.Random(args.seed)

    attempts = []
    if os.path.exists(log_path):
        attempts = json.load(open(log_path))
        print(f"Resuming from {len(attempts)} existing attempts")

    start_idx = len(attempts)
    for i in range(start_idx, args.n_samples):
        script_path = os.path.join(args.output_dir, f"iter_{i+1:03d}.py")
        code, params = sample_schedule(i, rng)
        with open(script_path, "w") as f:
            f.write(code)

        t0 = time.time()
        result = score_on_farm(script_path, args.train_problem, args.timeout, schedule_only=True)
        train_time = time.time() - t0

        entry = {
            "attempt": i + 1,
            "timestamp": time.time(),
            "params": params,
        }

        if "error" in result:
            entry["error"] = result["error"][:200]
            print(f"[{i+1}/{args.n_samples}] ERROR: {result['error'][:80]}")
        else:
            entry["train_aep"] = result.get("aep_gwh")
            entry["train_feasible"] = result.get("feasible")
            entry["train_time"] = round(train_time, 1)
            entry["train_baseline"] = result.get("baseline")
            print(f"[{i+1}/{args.n_samples}] AEP={result.get('aep_gwh', '?'):.1f}  "
                  f"feas={result.get('feasible', '?')}  ({train_time:.0f}s)")

            # ROWP (silently, like LLM runs)
            if not args.skip_rowp and os.path.exists(args.rowp_problem):
                rowp_result = score_on_farm(
                    script_path, args.rowp_problem, args.timeout, schedule_only=True)
                if "aep_gwh" in rowp_result:
                    entry["rowp_aep"] = rowp_result["aep_gwh"]
                    entry["rowp_feasible"] = rowp_result.get("feasible")

        attempts.append(entry)

        # Save incrementally
        with open(log_path, "w") as f:
            json.dump(attempts, f, indent=2)

    # Summary
    scored = [a for a in attempts if "train_aep" in a]
    rowp = [a for a in scored if "rowp_aep" in a]
    if scored:
        best_t = max(scored, key=lambda a: a["train_aep"])
        print(f"\n=== SUMMARY ===")
        print(f"Total: {len(attempts)}, scored: {len(scored)}, errors: {len(attempts)-len(scored)}")
        print(f"Best train: {best_t['train_aep']:.1f} GWh")
        if rowp:
            best_r = max(rowp, key=lambda a: a["rowp_aep"])
            print(f"Best ROWP:  {best_r['rowp_aep']:.1f} GWh (its train={best_r['train_aep']:.1f})")


if __name__ == "__main__":
    main()
