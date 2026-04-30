"""
HYPOTHESIS: Larger population with crossover between top 2 layouts and wind-aware targeted perturbations yields further gains over basic hybrid.
AXIS: hybrid_pop_sgd
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve
from pixwake.optim.sgd import boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):

    # ── Objective ──────────────────────────────────────────────────
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    diag = jnp.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)

    # ── Wind direction analysis ────────────────────────────────────
    wd_rad = wd * jnp.pi / 180.0
    u = jnp.cos(wd_rad)
    v = jnp.sin(wd_rad)
    mean_u = jnp.sum(u * weights)
    mean_v = jnp.sum(v * weights)
    dom_dir = jnp.arctan2(mean_v, mean_u)

    # ── Helper: filter inside polygon ──────────────────────────────
    n_verts = boundary.shape[0]
    def inside_mask(cand_x, cand_y):
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (cand_x - x1) * (-ey / el) + (cand_y - y1) * (ex / el)
        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0) > 0

    def generate_random_init(key):
        k1, k2 = jax.random.split(key)
        return (jax.random.uniform(k1, (n_target,), minval=float(x_min), maxval=float(x_max)),
                jax.random.uniform(k2, (n_target,), minval=float(y_min), maxval=float(y_max)))

    def ensure_inside(cand_x, cand_y, key):
        """Ensure we have n_target points inside polygon."""
        inside = inside_mask(cand_x, cand_y)
        ins_x, ins_y = cand_x[inside], cand_y[inside]
        n_inside = len(ins_x)
        if n_inside >= n_target:
            idx = jnp.round(jnp.linspace(0, n_inside - 1, n_target)).astype(int)
            return ins_x[idx], ins_y[idx]
        # Need more points — generate random and merge
        key, k1, k2 = jax.random.split(key, 3)
        rand_x, rand_y = generate_random_init(k1)
        combined_x = jnp.concatenate([ins_x, rand_x])
        combined_y = jnp.concatenate([ins_y, rand_y])
        return combined_x[:n_target], combined_y[:n_target]

    # ── Generate diverse foundation layouts ────────────────────────
    foundation = []
    key = jax.random.PRNGKey(42)

    # 1) Standard grid
    nx = max(3, int(jnp.ceil((x_max - x_min) / (min_spacing * 2.5))))
    ny = max(3, int(jnp.ceil((y_max - y_min) / (min_spacing * 2.5))))
    gx, gy = jnp.meshgrid(
        jnp.linspace(x_min + min_spacing, x_max - min_spacing, nx),
        jnp.linspace(y_min + min_spacing, y_max - min_spacing, ny))
    foundation.append((gx.flatten(), gy.flatten()))

    # 2) Wind-aligned grid (various spacings)
    perp_dir = dom_dir + jnp.pi / 2
    for s_factor in [6.0, 7.0, 8.0]:
        spacing_along = s_factor * min_spacing
        spacing_across = max(4.0, s_factor - 2.0) * min_spacing
        n_cols = max(3, int(diag / max(spacing_across, 1)))
        n_rows = max(3, int(diag / max(spacing_along, 1)))
        cdx, cdy = [], []
        for row in range(n_rows):
            for col in range(n_cols):
                offset = (0.5 * spacing_across) if row % 2 == 1 else 0.0
                dx = (col - n_cols / 2) * spacing_across + offset
                dy = (row - n_rows / 2) * spacing_along
                tx = cx + dx * jnp.cos(perp_dir) - dy * jnp.sin(perp_dir)
                ty = cy + dx * jnp.sin(perp_dir) + dy * jnp.cos(perp_dir)
                cdx.append(float(tx))
                cdy.append(float(ty))
        foundation.append((jnp.array(cdx), jnp.array(cdy)))

    # ── Helper: run SGD on a candidate ─────────────────────────────
    settings = SGDSettings(
        learning_rate=120.0,
        max_iter=2500,
        additional_constant_lr_iterations=1500,
        tol=1e-6,
        beta1=0.9,
        beta2=0.999,
        gamma_min_factor=0.001,
        ks_rho=100.0,
        spacing_weight=1.5,
        boundary_weight=2.0,
    )

    def run_sgd(init_x, init_y):
        return topfarm_sgd_solve(objective, init_x, init_y,
                                  boundary, min_spacing, settings)

    # ── Evaluate all foundation layouts ────────────────────────────
    population = []
    for fx, fy in foundation:
        key, k1 = jax.random.split(key)
        init_x, init_y = ensure_inside(fx, fy, k1)
        ox, oy = run_sgd(init_x, init_y)
        loss_val = float(objective(ox, oy)) + \
                   float(spacing_penalty(ox, oy, min_spacing)) + \
                   float(boundary_penalty(ox, oy, boundary))
        population.append((loss_val, ox, oy))

    # Sort by loss
    population.sort(key=lambda t: t[0])

    # Keep top 2
    top2 = population[:2]

    # ── Generate more candidates by mixing top layouts ─────────────
    for s in range(8):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        if s < 4:
            # Crossover between top 2 layouts
            p1_x, p1_y = top2[0][1], top2[0][2]
            p2_x, p2_y = top2[1][1], top2[1][2]
            swap_mask = jax.random.bernoulli(k1, p=0.5, shape=(n_target,))
            init_x = jnp.where(swap_mask, p1_x, p2_x)
            init_y = jnp.where(swap_mask, p1_y, p2_y)
        else:
            # Perturb top layout along wind direction
            ref_x, ref_y = top2[0][1], top2[0][2]
            perp_noise = jax.random.normal(k2, (n_target,)) * 0.05 * diag
            along_noise = jax.random.normal(k3, (n_target,)) * 0.02 * diag
            init_x = ref_x + along_noise * jnp.cos(dom_dir) - perp_noise * jnp.sin(dom_dir)
            init_y = ref_y + along_noise * jnp.sin(dom_dir) + perp_noise * jnp.cos(dom_dir)

        ox, oy = run_sgd(init_x, init_y)
        loss_val = float(objective(ox, oy)) + \
                   float(spacing_penalty(ox, oy, min_spacing)) + \
                   float(boundary_penalty(ox, oy, boundary))
        population.append((loss_val, ox, oy))
        population.sort(key=lambda t: t[0])
        top2 = population[:2]

    # Return best
    best = population[0]
    return best[1], best[2]