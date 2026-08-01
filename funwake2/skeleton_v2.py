"""FunWake-2 scale-aware skeleton (additive; leaves v1 skeleton untouched).

The frozen skeleton the evolved schedule sees. It differs from
``playground/skeleton.py`` in exactly two ways:

  1. **Signature.** The schedule is called as::

         schedule_fn(step, total_steps, D, min_spacing, n_turbines,
                     gamma_min, alpha0)

     i.e. the problem-intrinsic scales (rotor diameter ``D``, packing scale
     ``min_spacing``, count ``n_turbines``) and the user-supplied constraint
     tolerance ``gamma_min`` are passed through, instead of a driver-supplied
     ``lr0``. **There is NO hardcoded / driver ``lr0`` anywhere** — the
     exploration learning rate must be constructed from ``D`` *inside* the
     schedule (the structural fix for the v1 lr0 confound).

  2. **alpha0 normalization (D-2).** ``alpha0 = mean(|grad_x J|, |grad_y J|) / D``
     at the wind-aware initial layout — NOT the v1 driver default
     ``mean|grad J| / lr0``. At c = 0.833 those differ by 1.2x
     (``mean|grad J|/(0.833 D) = 1.2 * mean|grad J|/D``); shipping the ``/lr``
     form would put every ported penalty schedule 1.2x off. See G4.

Everything else — wind-aware grid init (single polygon) / zone-interior init
(multizone), the objective/constraint gradients, and the JIT-compiled Adam
update — is copied verbatim from the vetted v1 skeletons
(``playground/skeleton.py`` and ``parqo/skeleton_multizone.py``) so that a
faithful port lands bit-for-bit on the recorded baseline (G1).

``total_steps`` default = 8000 (decision D-1).
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# pixwake + the multizone helpers live in the source tree; import read-only.
for _p in (os.path.join(_ROOT, "dependencies", "pixwake", "src"),
           os.path.join(_ROOT, "parqo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pixwake.optim.sgd import boundary_penalty, spacing_penalty  # noqa: E402
# multizone boundary handling reused verbatim from the vetted Parque skeleton
from skeleton_multizone import (                                  # noqa: E402
    _init_positions, multizone_penalty,
)


def _adam_loop(grad_obj, grad_con, schedule_fn, x, y, D, min_spacing,
               n_turbines, gamma_min, alpha0, total_steps,
               early_stopping=False, es_threshold=0.1, max_lr=1.0):
    """JIT-compiled Adam loop — body identical to the v1 skeletons, only the
    schedule call signature is the scale-aware one."""

    @jax.jit
    def run_loop(x, y, alpha0):
        mx = jnp.zeros_like(x)
        my = jnp.zeros_like(y)
        vx = jnp.zeros_like(x)
        vy = jnp.zeros_like(y)
        eps = 1e-12

        def step(i, carry):
            x, y, mx, my, vx, vy = carry

            # scale-aware schedule controls lr, alpha, beta1, beta2
            lr, alpha, b1, b2 = schedule_fn(
                i, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0)

            gox, goy = grad_obj(x, y)
            gcx, gcy = grad_con(x, y)
            es_on = jnp.logical_and(early_stopping, lr <= es_threshold * max_lr)
            gox = jnp.where(es_on, 0.0, gox)
            goy = jnp.where(es_on, 0.0, goy)
            jx = gox + alpha * gcx
            jy = goy + alpha * gcy

            it = (i + 1).astype(float)
            mx_new = b1 * mx + (1 - b1) * jx
            my_new = b1 * my + (1 - b1) * jy
            vx_new = b2 * vx + (1 - b2) * jx**2
            vy_new = b2 * vy + (1 - b2) * jy**2

            mx_hat = mx_new / (1 - b1**it)
            my_hat = my_new / (1 - b1**it)
            vx_hat = vx_new / (1 - b2**it)
            vy_hat = vy_new / (1 - b2**it)

            x_new = x - lr * mx_hat / (jnp.sqrt(vx_hat) + eps)
            y_new = y - lr * my_hat / (jnp.sqrt(vy_hat) + eps)

            return (x_new, y_new, mx_new, my_new, vx_new, vy_new)

        init = (x, y, mx, my, vx, vy)
        final = jax.lax.fori_loop(0, total_steps, step, init)
        return final[0], final[1]

    return run_loop(x, y, alpha0)


def _wind_aware_init(boundary, min_spacing, wd, ws, weights, n_target, seed):
    """Single-polygon wind-aware grid init (verbatim from playground/skeleton.py
    and tools/reeval_lr0.py — the init that produced the recorded baselines)."""
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)

    wd_rad = jnp.deg2rad(wd)
    dominant = jnp.arctan2(
        jnp.sum(weights * jnp.sin(wd_rad)),
        jnp.sum(weights * jnp.cos(wd_rad)))
    angle = dominant + jnp.pi / 2

    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    cx, cy = jnp.mean(boundary[:, 0]), jnp.mean(boundary[:, 1])
    translated = boundary - jnp.array([cx, cy])
    rot = jnp.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_bnd = (rot @ translated.T).T

    rx_min, ry_min = jnp.min(rot_bnd, axis=0)
    rx_max, ry_max = jnp.max(rot_bnd, axis=0)
    nx = int(jnp.ceil((rx_max - rx_min) / min_spacing))
    ny = int(jnp.ceil((ry_max - ry_min) / min_spacing))
    gx, gy = jnp.meshgrid(
        jnp.linspace(rx_min + min_spacing / 2, rx_max - min_spacing / 2, nx),
        jnp.linspace(ry_min + min_spacing / 2, ry_max - min_spacing / 2, ny))
    rot_pts = jnp.stack([gx.flatten(), gy.flatten()], axis=-1)
    inv_rot = jnp.array([[cos_a, sin_a], [-sin_a, cos_a]])
    orig_pts = (inv_rot @ rot_pts.T).T + jnp.array([cx, cy])
    cand_x, cand_y = orig_pts[:, 0], orig_pts[:, 1]

    n_verts = boundary.shape[0]

    def edge_dist(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex, ey = x2 - x1, y2 - y1
        el = jnp.sqrt(ex**2 + ey**2) + 1e-10
        return (cand_x - x1) * (-ey / el) + (cand_y - y1) * (ex / el)

    inside = jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0) > 0
    ix, iy = cand_x[inside], cand_y[inside]

    if len(ix) >= n_target:
        key = jax.random.PRNGKey(seed)
        indices = jax.random.choice(key, len(ix), (n_target,), replace=False)
        return ix[indices], iy[indices]
    key = jax.random.PRNGKey(seed)
    x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
    key, _ = jax.random.split(key)
    y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))
    return x, y


def run_with_schedule(schedule_fn, sim, n_target, boundary, min_spacing,
                      wd, ws, weights, D, gamma_min,
                      total_steps=8000, seed=0, zones=None,
                      early_stopping=False, es_threshold=0.1):
    """Run the fixed scale-aware Adam skeleton with ``schedule_fn``.

    ``zones`` is None for a single-polygon farm (DEI/ROWP): the constraint is
    ``boundary_penalty + spacing_penalty`` and init is the wind-aware grid.
    When ``zones`` is a list of polygons (Parque multizone) the constraint is
    ``multizone_penalty + spacing_penalty`` and init is zone-interior.

    Returns (opt_x, opt_y).
    """
    D = float(D)

    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    if zones is None:
        def con_penalty(x, y):
            return (boundary_penalty(x, y, boundary)
                    + spacing_penalty(x, y, min_spacing))
    else:
        _zones = [jnp.asarray(z) for z in zones]

        def con_penalty(x, y):
            return (multizone_penalty(x, y, _zones)
                    + spacing_penalty(x, y, min_spacing))

    grad_obj = jax.grad(aep_objective, argnums=(0, 1))
    grad_con = jax.grad(con_penalty, argnums=(0, 1))

    # ── initialization ─────────────────────────────────────────────
    if zones is None:
        x, y = _wind_aware_init(
            boundary, min_spacing, wd, ws, weights, n_target, seed)
    else:
        x, y = _init_positions(zones, n_target, min_spacing, seed, init="zones")

    # ── alpha0 = mean|grad J| / D  (D-2; NOT /lr0) ─────────────────
    gox, goy = grad_obj(x, y)
    alpha0 = jnp.mean(jnp.abs(jnp.concatenate([gox, goy]))) / D
    # ── G8 (determinism): canonicalize alpha0 at the skeleton boundary ──────
    # AEP is pathologically sensitive to alpha0 in the chaotic low-lr tail (a
    # ~5th-significant-figure change moves AEP ~1.5 GWh). float64 reduction order
    # can differ ~1 ULP across machines/XLA builds, which then explodes. Round-
    # tripping through float32 pins alpha0 to a canonical value every environment
    # agrees on (float64 reductions agree to ~14 sig figs >> float32's ~7), so the
    # same (schedule, cell, seed) is bit-reproducible across fresh processes and
    # machines. Computed once, outside the JIT loop — zero hot-path cost.
    alpha0 = float(np.float32(float(alpha0)))

    # peak lr (only needed for the optional post-hoc early-stopping trigger)
    max_lr = 1.0
    if early_stopping:
        steps = jnp.arange(total_steps)
        lrs = jax.vmap(lambda i: schedule_fn(
            i, total_steps, D, min_spacing, n_target, gamma_min, alpha0)[0])(steps)
        max_lr = jnp.max(lrs)

    return _adam_loop(
        grad_obj, grad_con, schedule_fn, x, y, D, min_spacing, n_target,
        gamma_min, alpha0, total_steps,
        early_stopping=early_stopping, es_threshold=es_threshold, max_lr=max_lr)


def compute_alpha0_and_gradnorm(sim, n_target, boundary, min_spacing,
                                wd, ws, weights, D, seed=0, zones=None):
    """Return (mean_grad_norm, alpha0=mean_grad_norm/D) at the init layout —
    used by the G4 alpha0 gate to show the /D vs /lr numeric relationship."""
    D = float(D)

    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    grad_obj = jax.grad(aep_objective, argnums=(0, 1))
    if zones is None:
        x, y = _wind_aware_init(
            boundary, min_spacing, wd, ws, weights, n_target, seed)
    else:
        x, y = _init_positions(zones, n_target, min_spacing, seed, init="zones")
    gox, goy = grad_obj(x, y)
    gnorm = float(jnp.mean(jnp.abs(jnp.concatenate([gox, goy]))))
    return gnorm, gnorm / D
