"""Stochastic AEP estimator for Quick-2023-style SGD.

Two paths over the same Weibull-per-sector resource on the DEI training farm:
  A. Deterministic fine-grid AEP: fine wd × ws quadrature, weighted by per-sector
     Weibull pdf and sector_probability. Used as ground truth for unbiasedness.
  B. Stochastic K-sample MC estimator: each call samples K (wd, ws) pairs by
     (i) categorical sector draw with sector_probability, (ii) uniform draw
     within the sector for wd, (iii) inverse-CDF Weibull(A_sector, k_sector)
     for ws. AEP_hat = 8760 * mean(P_total)/1e6  in GWh.

Sanity check: averaging many MC estimates of (B) should converge to (A).

Wake model held identical to iter_192's original discovery: pixwake's
BastankhahGaussianDeficit(k=0.04). The skeleton signature is unchanged.

Usage:
    PYTHONPATH=dependencies/pixwake/src pixi run python \\
      validation/stochastic_aep/stochastic_aep.py \\
      --resource validation/stochastic_aep/dei_weibull_12.json \\
      --problem playground/problem.json \\
      --K 50 --repeats 200 \\
      --out validation/stochastic_aep/stochastic_aep_dei.json
"""
import argparse
import json
import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.deficit.noj import NOJDeficit
from pixwake.superposition import SquaredSum


def build_sim(problem, wake_model="bastankhah_0.04"):
    """Build pixwake WakeSimulation from a problem-style dict.

    wake_model:
      - 'bastankhah_0.04' (matches iter_192 discovery setup, default)
      - 'noj_0.05'        (matches IEA 740-10 published setup)
    """
    D = problem["rotor_diameter"]
    hub_height = problem.get("hub_height", 150.0)
    t = problem.get("turbine", problem)
    ws_arr = jnp.array(t["power_curve_ws"], dtype=float)
    power = jnp.array(t["power_curve_kw"], dtype=float)
    ct_ws = jnp.array(t.get("ct_curve_ws", t["power_curve_ws"]), dtype=float)
    ct = jnp.array(t["ct_curve_ct"], dtype=float)
    turbine = Turbine(
        rotor_diameter=D, hub_height=hub_height,
        power_curve=Curve(ws=ws_arr, values=power),
        ct_curve=Curve(ws=ct_ws, values=ct),
    )
    if wake_model == "bastankhah_0.04":
        deficit = BastankhahGaussianDeficit(k=0.04)
    elif wake_model == "noj_0.05":
        deficit = NOJDeficit(k=0.05, superposition=SquaredSum())
    else:
        raise ValueError(f"unknown wake_model: {wake_model}")
    sim = WakeSimulation(turbine, deficit)
    return sim, D


def wind_aware_grid_init(boundary, min_spacing, n_target, seed=0):
    """Same wind-aware grid init the skeleton uses. Returns (x, y) jnp arrays."""
    boundary = jnp.array(boundary)
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    cx, cy = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])
    # No wind direction given — just axis-aligned grid
    rx_min, ry_min = x_min - cx, y_min - cy
    rx_max, ry_max = x_max - cx, y_max - cy
    nx = int(jnp.ceil((rx_max - rx_min) / min_spacing))
    ny = int(jnp.ceil((ry_max - ry_min) / min_spacing))
    gx, gy = jnp.meshgrid(
        jnp.linspace(rx_min + min_spacing/2, rx_max - min_spacing/2, nx),
        jnp.linspace(ry_min + min_spacing/2, ry_max - min_spacing/2, ny),
    )
    cand_x, cand_y = gx.flatten() + cx, gy.flatten() + cy

    # Inside-boundary mask
    n_verts = boundary.shape[0]
    inside = jnp.ones_like(cand_x, dtype=bool)
    for i in range(n_verts):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = jnp.sqrt(ex**2 + ey**2) + 1e-10
        dist = (cand_x - x1) * (-ey / el) + (cand_y - y1) * (ex / el)
        inside &= dist > 0
    ix, iy = cand_x[inside], cand_y[inside]
    key = jax.random.PRNGKey(seed)
    if len(ix) >= n_target:
        indices = jax.random.choice(key, len(ix), (n_target,), replace=False)
        return ix[indices], iy[indices]
    # Fallback: uniform inside bbox
    k1, k2 = jax.random.split(key)
    x = jax.random.uniform(k1, (n_target,), minval=float(x_min), maxval=float(x_max))
    y = jax.random.uniform(k2, (n_target,), minval=float(y_min), maxval=float(y_max))
    return x, y


def deterministic_fine_grid_aep(sim, x, y, resource, ws_min=4.0, ws_max=25.0,
                                ws_step=1.0, wd_step=1.0):
    """Compute AEP over a fine (wd, ws) grid with Weibull pdf weights.

    For each fine wd_i (1 deg bin in [0,360)):
        identify the sector j it belongs to (nearest sector center, modulo 360),
        compute prob(ws_k) = weibull_pdf(ws_k; A_j, k_j) * ws_step,
        weight by sector_probability[j] * (1 / sector_width_in_fine_bins).
    """
    centers = np.array(resource["sector_centers_deg"])
    sector_width = float(resource["sector_width_deg"])
    n_sectors = int(resource["n_sectors"])
    A = np.array(resource["weibull_A"])
    k = np.array(resource["weibull_k"])
    freq = np.array(resource["sector_probability"])

    wd_grid = np.arange(0.0, 360.0, wd_step)
    ws_grid = np.arange(ws_min, ws_max + ws_step, ws_step)
    n_wd = len(wd_grid)
    n_ws = len(ws_grid)

    # For each fine wd_i, assign its sector index
    shifted = (wd_grid + sector_width / 2) % 360.0
    sec_idx = (shifted // sector_width).astype(int) % n_sectors

    # Number of fine bins per sector
    bins_per_sector = round(sector_width / wd_step)

    # Build joint probability matrix shape (n_wd, n_ws)
    P = np.zeros((n_wd, n_ws))
    for i, w in enumerate(wd_grid):
        s = sec_idx[i]
        A_s, k_s = A[s], k[s]
        # Weibull pdf at each ws bin
        pdf = (k_s / A_s) * (ws_grid / A_s)**(k_s - 1) * np.exp(-(ws_grid / A_s)**k_s)
        P[i, :] = freq[s] / bins_per_sector * pdf * ws_step

    P_flat = jnp.array(P.flatten())
    ws_flat = jnp.array(np.tile(ws_grid, n_wd))
    wd_flat = jnp.array(np.repeat(wd_grid, n_ws))
    ti_flat = jnp.full_like(ws_flat, 0.06)

    r = sim(x, y, ws_amb=ws_flat, wd_amb=wd_flat, ti_amb=ti_flat)
    return float(r.aep(probabilities=P_flat))


def stochastic_aep_factory(sim, resource, ws_min=4.0, ws_max=25.0):
    """Build a JIT-friendly stochastic AEP estimator over the turbine's
    operational envelope [ws_min, ws_max].

    Returns:
        aep_fn(x, y, key, K) -> AEP estimate in GWh, unbiased of the truncated
        Weibull-marginalized AEP integral over ws ∈ [ws_min, ws_max].

    Sampling:
      sector ~ Categorical(sector_probability)
      wd ~ Uniform(center - width/2, center + width/2)
      u ~ Uniform(0, 1)
      ws = A_sector * (-log(1 - u))^(1/k_sector)   (inverse CDF Weibull)
    Then samples outside [ws_min, ws_max] are zero-weighted (turbine power
    extrapolation is invalid there, so we mask them rather than rejecting,
    which keeps shape static for JIT).
    """
    centers = jnp.array(resource["sector_centers_deg"])
    sector_width = float(resource["sector_width_deg"])
    A = jnp.array(resource["weibull_A"])
    k = jnp.array(resource["weibull_k"])
    freq = jnp.array(resource["sector_probability"])
    n_sectors = int(resource["n_sectors"])
    ws_min = float(ws_min)
    ws_max = float(ws_max)

    def aep_fn(x, y, key, K):
        k1, k2, k3 = jax.random.split(key, 3)
        sec_idx = jax.random.choice(k1, n_sectors, (K,), p=freq)
        wd_off = jax.random.uniform(k2, (K,), minval=-sector_width / 2, maxval=sector_width / 2)
        wd_samp = (centers[sec_idx] + wd_off) % 360.0
        u = jax.random.uniform(k3, (K,))
        A_s = A[sec_idx]
        k_s = k[sec_idx]
        ws_samp = A_s * (-jnp.log(1.0 - u))**(1.0 / k_s)
        # Clamp out-of-range samples to interior so the turbine power-curve
        # interpolation is well-defined; mask their contribution to AEP to 0.
        in_range = (ws_samp >= ws_min) & (ws_samp <= ws_max)
        ws_eval = jnp.clip(ws_samp, ws_min, ws_max)
        ti_samp = jnp.full_like(ws_samp, 0.06)
        r = sim(x, y, ws_amb=ws_eval, wd_amb=wd_samp, ti_amb=ti_samp)
        # probability 1/K per sample × in_range mask. Note this is an
        # unbiased estimator of the truncated integral
        # P(ws ∈ [ws_min, ws_max]) × E[P_total | ws ∈ [ws_min, ws_max]] × 8760,
        # which equals ∫_{ws ∈ [ws_min, ws_max]} P_total f(ws) dws — the same
        # quantity the deterministic Riemann sum approximates.
        probs = (1.0 / K) * in_range.astype(ws_samp.dtype)
        return r.aep(probabilities=probs)

    return aep_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resource", required=True)
    p.add_argument("--problem", required=True)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--repeats", type=int, default=200, help="Number of independent MC estimates to average for unbiasedness check")
    p.add_argument("--out", required=True)
    p.add_argument("--n-cost-iters", type=int, default=200, help="Iterations to time for per-iter cost (warm)")
    p.add_argument("--wake-model", default="bastankhah_0.04",
                   choices=["bastankhah_0.04", "noj_0.05"],
                   help="Wake model. bastankhah_0.04 matches iter_192 discovery (DEI); noj_0.05 matches 740-10 published")
    p.add_argument("--ws-step", type=float, default=1.0,
                   help="Wind-speed step (m/s) for deterministic fine-grid AEP. Smaller = more accurate Riemann sum.")
    args = p.parse_args()

    with open(args.resource) as f:
        resource = json.load(f)
    with open(args.problem) as f:
        problem = json.load(f)

    sim, D = build_sim(problem, wake_model=args.wake_model)
    boundary = problem["boundary_vertices"]
    min_spacing = float(problem["min_spacing_m"])
    n_target = int(problem["n_target"])
    x, y = wind_aware_grid_init(boundary, min_spacing, n_target, seed=0)

    # Deterministic fine-grid AEP under the SAME Weibull resource
    t0 = time.time()
    det_aep = deterministic_fine_grid_aep(sim, x, y, resource,
                                           ws_min=4.0, ws_max=25.0,
                                           ws_step=args.ws_step, wd_step=1.0)
    t_det = time.time() - t0

    # Stochastic AEP estimator
    aep_fn = stochastic_aep_factory(sim, resource)
    aep_jit = jax.jit(aep_fn, static_argnames=("K",))

    # Warm-up JIT
    key = jax.random.PRNGKey(0)
    _ = float(aep_jit(x, y, key, args.K))

    # Unbiasedness check: average over many independent runs
    repeats = args.repeats
    ests = np.zeros(repeats)
    t0 = time.time()
    for i in range(repeats):
        key, subkey = jax.random.split(key)
        ests[i] = float(aep_jit(x, y, subkey, args.K))
    t_rep = time.time() - t0
    mc_mean = float(np.mean(ests))
    mc_se = float(np.std(ests, ddof=1) / np.sqrt(repeats))
    mc_std = float(np.std(ests, ddof=1))

    # Per-iter cost (forward pass only; gradient adds ~2x)
    t0 = time.time()
    for i in range(args.n_cost_iters):
        key, subkey = jax.random.split(key)
        _ = float(aep_jit(x, y, subkey, args.K))
    t_per_iter = (time.time() - t0) / args.n_cost_iters

    # Per-iter cost with gradient (this is the SGD-relevant number)
    grad_fn = jax.grad(lambda xv, yv, key, K: aep_fn(xv, yv, key, K), argnums=(0, 1))
    grad_jit = jax.jit(grad_fn, static_argnames=("K",))
    # Warm-up
    key, subkey = jax.random.split(key)
    gx, gy = grad_jit(x, y, subkey, args.K)
    _ = float(jnp.sum(gx))
    t0 = time.time()
    for i in range(args.n_cost_iters):
        key, subkey = jax.random.split(key)
        gx, gy = grad_jit(x, y, subkey, args.K)
        _ = float(jnp.sum(gx) + jnp.sum(gy))
    t_per_iter_grad = (time.time() - t0) / args.n_cost_iters

    # Project candidates/hour: assume 8000 SGD iters per candidate
    iters_per_candidate = 8000
    sec_per_candidate = t_per_iter_grad * iters_per_candidate
    candidates_per_hour = 3600 / sec_per_candidate if sec_per_candidate > 0 else float("inf")

    out = {
        "config": {
            "problem": args.problem,
            "resource": args.resource,
            "K": args.K,
            "repeats": args.repeats,
            "n_target": n_target,
            "min_spacing_m": min_spacing,
            "wake_model": args.wake_model,
        },
        "deterministic_fine_grid_aep_gwh": det_aep,
        "deterministic_eval_time_s": t_det,
        "stochastic_K_sample_estimator": {
            "K": args.K,
            "repeats": repeats,
            "mc_mean_aep_gwh": mc_mean,
            "mc_std_aep_gwh": mc_std,
            "mc_se_aep_gwh": mc_se,
            "delta_vs_deterministic_gwh": mc_mean - det_aep,
            "delta_vs_deterministic_pct": (mc_mean - det_aep) / det_aep * 100,
            "z_score": (mc_mean - det_aep) / mc_se if mc_se > 0 else float("nan"),
        },
        "per_iter_cost": {
            "forward_only_s": t_per_iter,
            "forward_plus_grad_s": t_per_iter_grad,
            "iters_per_candidate_assumed": iters_per_candidate,
            "sec_per_candidate": sec_per_candidate,
            "candidates_per_hour": candidates_per_hour,
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
