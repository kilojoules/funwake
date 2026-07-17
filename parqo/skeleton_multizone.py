"""Multi-zone variant of the fixed optimization skeleton
(playground/skeleton.py). Same Adam loop and schedule_fn contract; only
the boundary handling differs: the boundary is a set of disconnected
inclusion-zone polygons (Criado Risco et al. 2024, WES 9, 585-600).

Boundary penalty: per turbine, signed distance to the NEAREST inclusion
zone (min over zone SDFs, negative inside), penalized as
sum(relu(sdf)^2) — the multi-polygon analogue of
pixwake.optim.boundary.containment_penalty(convex=False).
"""
import jax
import jax.numpy as jnp
import numpy as np
from pixwake.optim.boundary import polygon_sdf
from pixwake.optim.sgd import spacing_penalty


def multizone_sdf(x, y, zones):
    """Signed distance to nearest inclusion zone (negative inside one)."""
    return jnp.min(jnp.stack([polygon_sdf(x, y, z) for z in zones]), axis=0)


def multizone_penalty(x, y, zones):
    sdf = multizone_sdf(x, y, zones)
    return jnp.sum(jnp.maximum(0.0, sdf) ** 2)


def _init_positions(zones, n_target, min_spacing, seed, init="zones"):
    """init="zones": candidate grid filtered to zone interiors, randomly
    subsampled (area-weighted via candidate density).
    init="box": regular grid over the zone bounding box, ignoring zones
    (turbines may start outside inclusion zones, as in the paper's
    baseline where the solver pulls them to the nearest polygon)."""
    from shapely.geometry import Point, Polygon

    allb = np.vstack([np.asarray(z) for z in zones])
    x_min, y_min = allb.min(axis=0)
    x_max, y_max = allb.max(axis=0)
    rng = np.random.default_rng(seed)

    if init == "box":
        aspect = (x_max - x_min) / (y_max - y_min)
        ny = int(np.ceil(np.sqrt(n_target / aspect)))
        nx = int(np.ceil(n_target / ny))
        gx = np.linspace(x_min, x_max, nx)
        gy = np.linspace(y_min, y_max, ny)
        GX, GY = np.meshgrid(gx, gy)
        pts = np.c_[GX.ravel(), GY.ravel()]
        idx = rng.choice(len(pts), n_target, replace=False) \
            if len(pts) > n_target else np.arange(len(pts))
        sel = pts[idx]
        # tiny jitter so no two turbines are exactly coincident
        sel = sel + rng.normal(0, 1.0, sel.shape)
        return jnp.array(sel[:, 0]), jnp.array(sel[:, 1])

    polys = [Polygon(np.asarray(z)) for z in zones]
    pitch = min_spacing / 2.0
    gx = np.arange(x_min + pitch / 2, x_max, pitch)
    gy = np.arange(y_min + pitch / 2, y_max, pitch)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.c_[GX.ravel(), GY.ravel()]
    inside = np.array([any(p.contains(Point(q)) for p in polys) for q in pts])
    cand = pts[inside]

    if len(cand) >= n_target:
        idx = rng.choice(len(cand), n_target, replace=False)
        sel = cand[idx]
    else:
        extra = rng.uniform([x_min, y_min], [x_max, y_max],
                            (n_target - len(cand), 2))
        sel = np.vstack([cand, extra])
    return jnp.array(sel[:, 0]), jnp.array(sel[:, 1])


def run_with_schedule(schedule_fn, sim, n_target, zones, min_spacing,
                      wd, ws, weights, total_steps=8000, seed=0,
                      init="zones"):
    """Fixed Adam skeleton with multi-zone boundary. Returns (x, y)."""
    zones = [jnp.asarray(z) for z in zones]

    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    def con_penalty(x, y):
        return (multizone_penalty(x, y, zones)
                + spacing_penalty(x, y, min_spacing))

    grad_obj = jax.grad(aep_objective, argnums=(0, 1))
    grad_con = jax.grad(con_penalty, argnums=(0, 1))

    x, y = _init_positions(zones, n_target, min_spacing, seed, init=init)

    gox, goy = grad_obj(x, y)
    lr0 = 50.0
    alpha0 = jnp.mean(jnp.abs(jnp.concatenate([gox, goy]))) / lr0

    @jax.jit
    def run_loop(x, y):
        mx = jnp.zeros_like(x)
        my = jnp.zeros_like(y)
        vx = jnp.zeros_like(x)
        vy = jnp.zeros_like(y)
        eps = 1e-12

        def step(i, carry):
            x, y, mx, my, vx, vy = carry
            lr, alpha, b1, b2 = schedule_fn(i, total_steps, lr0, alpha0)

            gox, goy = grad_obj(x, y)
            gcx, gcy = grad_con(x, y)
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

    return run_loop(x, y)
