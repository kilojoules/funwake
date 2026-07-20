"""TopFarm basin finding followed by exact SLSQP polish.

HYPOTHESIS: Hex/wind-aware TopFarm SGD reaches better basins than direct
SLSQP starts, while SLSQP can still improve the final active-constraint polish.
AXIS: scipy_slsqp polish after short TopFarm SGD multistart basin search.
LESSON: Pending score.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from pixwake.optim.sgd import SGDSettings, boundary_penalty, spacing_penalty, topfarm_sgd_solve


def _edge_clearance(x, y, boundary):
    n_verts = boundary.shape[0]

    def edge(i):
        x1, y1 = boundary[i]
        x2, y2 = boundary[(i + 1) % n_verts]
        ex = x2 - x1
        ey = y2 - y1
        el = jnp.sqrt(ex * ex + ey * ey) + 1e-12
        return (x - x1) * (-ey / el) + (y - y1) * (ex / el)

    return jax.vmap(edge)(jnp.arange(n_verts))


def _feasible(x, y, boundary, min_spacing):
    n = x.shape[0]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = jnp.sqrt(dx * dx + dy * dy + jnp.eye(n) * 1e18)
    return (
        (boundary_penalty(x, y, boundary) < 1e-3)
        & (spacing_penalty(x, y, min_spacing) < 1e-3)
        & (jnp.min(dist) >= min_spacing * 0.999)
    )


def _candidates(boundary, min_spacing, angle, phase_x, phase_y, dense):
    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    center = jnp.mean(boundary, axis=0)
    span_x = x_max - x_min
    span_y = y_max - y_min
    diag = jnp.sqrt(span_x * span_x + span_y * span_y)
    step = min_spacing * (0.62 if dense else 1.04)
    row = step * 0.8660254037844386
    n_side = int(np.ceil(float(diag / row))) + 8
    ii, jj = jnp.meshgrid(
        jnp.arange(n_side) - n_side // 2,
        jnp.arange(n_side) - n_side // 2,
    )
    hx = (ii.ravel() + 0.5 * (jj.ravel() % 2) + phase_x) * step
    hy = (jj.ravel() + phase_y) * row
    ca = jnp.cos(angle)
    sa = jnp.sin(angle)
    x = center[0] + hx * ca - hy * sa
    y = center[1] + hx * sa + hy * ca
    clearance = jnp.min(_edge_clearance(x, y, boundary), axis=0)
    keep = clearance > min_spacing * (0.04 if dense else 0.08)
    return x[keep], y[keep]


def _farthest(cand_x, cand_y, n_target, boundary, mode):
    center = jnp.mean(boundary, axis=0)
    radial = (cand_x - center[0]) ** 2 + (cand_y - center[1]) ** 2
    first = jnp.where(mode == 0, jnp.argmax(radial), jnp.argmin(cand_y))
    out_x = jnp.zeros(n_target)
    out_y = jnp.zeros(n_target)
    out_x = out_x.at[0].set(cand_x[first])
    out_y = out_y.at[0].set(cand_y[first])
    best_d2 = (cand_x - out_x[0]) ** 2 + (cand_y - out_y[0]) ** 2
    for i in range(1, n_target):
        nxt = jnp.argmax(best_d2)
        out_x = out_x.at[i].set(cand_x[nxt])
        out_y = out_y.at[i].set(cand_y[nxt])
        d2 = (cand_x - out_x[i]) ** 2 + (cand_y - out_y[i]) ** 2
        best_d2 = jnp.minimum(best_d2, d2)
    return out_x, out_y


def _ordered(cand_x, cand_y, n_target):
    order = jnp.lexsort((cand_x, cand_y))
    idx = jnp.round(jnp.linspace(0, len(cand_x) - 1, n_target)).astype(int)
    return cand_x[order][idx], cand_y[order][idx]


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    def objective_xy(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :n_target]
        return -jnp.sum(p * weights[:, None]) * 8760.0 / 1e6

    @jax.jit
    def objective(coords):
        return objective_xy(coords[:n_target], coords[n_target:])

    grad_objective = jax.jit(jax.grad(objective))

    @jax.jit
    def boundary_constraints(coords):
        return _edge_clearance(coords[:n_target], coords[n_target:], boundary).T.ravel()

    boundary_jac = jax.jit(jax.jacobian(boundary_constraints))

    @jax.jit
    def spacing_constraints(coords):
        x = coords[:n_target]
        y = coords[n_target:]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx * dx + dy * dy
        iu, ju = jnp.triu_indices(n_target, k=1)
        return d2[iu, ju] - min_spacing * min_spacing

    spacing_jac = jax.jit(jax.jacobian(spacing_constraints))

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    wd_rad = jnp.deg2rad(wd)
    energy = weights * ws**3
    dom = jnp.arctan2(jnp.sum(jnp.sin(wd_rad) * energy), jnp.sum(jnp.cos(wd_rad) * energy))

    starts = []
    for angle, px, py, dense, mode in (
        (0.0, 0.0, 0.0, False, 2),
        (dom, 0.2, 0.35, False, 0),
        (dom + jnp.pi / 5.0, 0.45, 0.1, True, 1),
    ):
        cx, cy = _candidates(boundary, min_spacing, angle, px, py, dense)
        if len(cx) >= n_target:
            starts.append(_ordered(cx, cy, n_target) if mode == 2 else _farthest(cx, cy, n_target, boundary, mode))

    if not starts:
        idx = jnp.arange(n_target, dtype=jnp.float64)
        starts.append((x_min + ((idx * 0.61803398875) % 1.0) * (x_max - x_min),
                       y_min + ((idx * 0.75487766625) % 1.0) * (y_max - y_min)))

    settings = SGDSettings(
        learning_rate=180.0,
        max_iter=2600 if n_target <= 55 else 1800,
        additional_constant_lr_iterations=1200 if n_target <= 55 else 700,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        gamma_min_factor=0.01,
        ks_rho=100.0,
        spacing_weight=3.0,
        boundary_weight=2.5,
    )

    best_x, best_y = starts[0]
    best_obj = jnp.where(_feasible(best_x, best_y, boundary, min_spacing), objective_xy(best_x, best_y), jnp.inf)
    n_sgd = 2 if n_target <= 55 else 1
    for sx, sy in starts[:n_sgd]:
        ox, oy = topfarm_sgd_solve(objective_xy, sx, sy, boundary, min_spacing, settings)
        oobj = objective_xy(ox, oy)
        if _feasible(ox, oy, boundary, min_spacing) & (oobj < best_obj):
            best_x = ox
            best_y = oy
            best_obj = oobj
        sobj = objective_xy(sx, sy)
        if _feasible(sx, sy, boundary, min_spacing) & (sobj < best_obj):
            best_x = sx
            best_y = sy
            best_obj = sobj

    def scipy_obj(coords):
        return float(objective(jnp.asarray(coords)))

    def scipy_grad(coords):
        return np.asarray(grad_objective(jnp.asarray(coords)), dtype=np.float64)

    def scipy_boundary(coords):
        return np.asarray(boundary_constraints(jnp.asarray(coords)), dtype=np.float64)

    def scipy_boundary_jac(coords):
        return np.asarray(boundary_jac(jnp.asarray(coords)), dtype=np.float64)

    def scipy_spacing(coords):
        return np.asarray(spacing_constraints(jnp.asarray(coords)), dtype=np.float64)

    def scipy_spacing_jac(coords):
        return np.asarray(spacing_jac(jnp.asarray(coords)), dtype=np.float64)

    if n_target <= 55 and jnp.isfinite(best_obj):
        res = minimize(
            scipy_obj,
            np.asarray(jnp.concatenate([best_x, best_y]), dtype=np.float64),
            jac=scipy_grad,
            method="SLSQP",
            constraints=(
                {"type": "ineq", "fun": scipy_boundary, "jac": scipy_boundary_jac},
                {"type": "ineq", "fun": scipy_spacing, "jac": scipy_spacing_jac},
            ),
            bounds=[(float(x_min), float(x_max))] * n_target + [(float(y_min), float(y_max))] * n_target,
            options={"maxiter": 80, "ftol": 5e-7, "disp": False},
        )
        rx = jnp.asarray(res.x[:n_target])
        ry = jnp.asarray(res.x[n_target:])
        robj = objective(jnp.asarray(res.x))
        if _feasible(rx, ry, boundary, min_spacing) & (robj < best_obj):
            best_x = rx
            best_y = ry

    return jnp.asarray(best_x), jnp.asarray(best_y)
