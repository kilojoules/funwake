"""
HYPOTHESIS: A multi-sector initializer that places turbines in ranks perpendicular to 
  each dominant wind direction, then ensembles sector-specific layouts via softmax-weighted 
  consensus, yields better starting points than staggered grid, enabling AEP > 5553.29 GWh.
AXIS: wind_direction_voting
LESSON: jax_compiled_loop: 5552.35 feasible but below best. Compiled fori_loop underperforms 
  topfarm_sgd_solve's internal Adam scheduling.
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    def get_aep(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return jnp.sum(p * weights[:, None]) * 8760 / 1e6

    x_min, y_min = jnp.min(boundary, axis=0)
    x_max, y_max = jnp.max(boundary, axis=0)
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0

    n_verts = boundary.shape[0]
    def inside_mask(px, py):
        def edge_dist(i):
            x1, y1 = boundary[i]
            x2, y2 = boundary[(i + 1) % n_verts]
            ex, ey = x2 - x1, y2 - y1
            el = jnp.sqrt(ex**2 + ey**2) + 1e-10
            return (px - x1) * (-ey / el) + (py - y1) * (ex / el)
        return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts)), axis=0) > 0

    # ── Bin wind directions into 6 sectors ────────────────────────────
    wd_rad = wd * jnp.pi / 180.0
    n_sectors = 6
    sector_size = 360.0 / n_sectors

    sector_weights_list = []
    sector_mean_dirs = []

    for s in range(n_sectors):
        lo = s * sector_size
        hi = (s + 1) * sector_size
        # Handle wrap-around
        if hi <= 360:
            in_sector = (wd >= lo) & (wd < hi)
        else:
            in_sector = (wd >= lo) | (wd < (hi - 360))

        sector_w = weights * in_sector
        total_w = jnp.sum(sector_w)
        sector_weights_list.append(total_w)

        if total_w > 0:
            sector_wd = wd_rad[in_sector]
            sec_w = weights[in_sector] / total_w
            mean_sin = jnp.sum(sec_w * jnp.sin(sector_wd))
            mean_cos = jnp.sum(sec_w * jnp.cos(sector_wd))
            sector_mean_dirs.append(jnp.arctan2(mean_sin, mean_cos))
        else:
            sector_mean_dirs.append(jnp.array(0.0))

    sector_weights = jnp.array(sector_weights_list)
    sector_dirs = jnp.array(sector_mean_dirs)

    # Sort sectors by total weight (descending), keep top 4
    top4_idx = jnp.argsort(sector_weights)[::-1][:4]
    top4_w = sector_weights[top4_idx]
    top4_dir = sector_dirs[top4_idx]
    # Normalize for sampling
    top4_w = top4_w / jnp.sum(top4_w)

    # ── Generate sector-specific staggered grids ──────────────────────
    def sector_grid(dom_dir, offset_val):
        perp_x = jnp.cos(dom_dir + jnp.pi / 2)
        perp_y = jnp.sin(dom_dir + jnp.pi / 2)
        par_x = jnp.cos(dom_dir)
        par_y = jnp.sin(dom_dir)

        row_spacing = min_spacing * 2.0
        turb_spacing = min_spacing * 1.05
        n_perp = max(int(jnp.ceil((x_max - x_min) / turb_spacing)) + 4, 2)
        n_par = max(int(jnp.ceil((y_max - y_min) / row_spacing)) + 4, 2)

        pts_x, pts_y = [], []
        for ri in range(n_par):
            for ci in range(n_perp):
                offs = (ri % 2) * 0.5 * turb_spacing
                pp = (ci - n_perp / 2 + 0.5) * turb_spacing + offs
                pa = (ri - n_par / 2 + 0.5) * row_spacing
                pts_x.append(cx + pp * perp_x + pa * par_x)
                pts_y.append(cy + pp * perp_y + pa * par_y)

        pts_x = jnp.array(pts_x)
        pts_y = jnp.array(pts_y)
        inside = inside_mask(pts_x, pts_y)
        ix, iy = pts_x[inside], pts_y[inside]

        if len(ix) >= n_target:
            raw_idx = jnp.linspace(0, len(ix) - 1, n_target) + offset_val
            idx = jnp.clip(jnp.round(raw_idx).astype(int), 0, len(ix) - 1)
            return ix[idx], iy[idx]
        else:
            spacing = min_spacing * 1.05
            nx = int(jnp.ceil((x_max - x_min) / spacing))
            ny = int(jnp.ceil((y_max - y_min) / spacing))
            gx, gy = jnp.meshgrid(
                jnp.linspace(x_min + spacing / 2, x_max - spacing / 2, max(nx, 2)),
                jnp.linspace(y_min + spacing / 2, y_max - spacing / 2, max(ny, 2)),
            )
            cand_x, cand_y = gx.flatten(), gy.flatten()
            inside2 = inside_mask(cand_x, cand_y)
            ix2, iy2 = cand_x[inside2], cand_y[inside2]
            if len(ix2) >= n_target:
                raw_idx2 = jnp.linspace(0, len(ix2) - 1, n_target) + offset_val
                idx2 = jnp.clip(jnp.round(raw_idx2).astype(int), 0, len(ix2) - 1)
                return ix2[idx2], iy2[idx2]
            else:
                key = jax.random.PRNGKey(int(offset_val * 100) + int(dom_dir * 100))
                tx = jax.random.uniform(key, (n_target,),
                                        minval=float(x_min), maxval=float(x_max))
                key = jax.random.split(key)[1]
                ty = jax.random.uniform(key, (n_target,),
                                        minval=float(y_min), maxval=float(y_max))
                return tx, ty

    # ── Run ensemble: mix of sector-specific starts ───────────────────
    settings = SGDSettings(
        learning_rate=50.0,
        max_iter=2800,
        additional_constant_lr_iterations=1200,
        tol=1e-6,
        beta1=0.9,
        beta2=0.999,
        gamma_min_factor=0.01,
        ks_rho=100.0,
        spacing_weight=0.3,
        boundary_weight=0.3,
    )

    best_aep_sofar = -jnp.inf
    best_x_sofar = jnp.zeros(n_target)
    best_y_sofar = jnp.zeros(n_target)

    n_stored = 6
    stored_x = jnp.zeros((n_stored, n_target))
    stored_y = jnp.zeros((n_stored, n_target))
    stored_aep = jnp.zeros(n_stored) - 1e10

    # 2 runs per sector × 4 sectors = 8 runs
    n_per_sector = 2
    for si in range(4):
        dom_dir = top4_dir[si]
        for off_idx in range(n_per_sector):
            offset_val = float(off_idx) * 0.5
            init_x, init_y = sector_grid(dom_dir, offset_val)
            trial_x, trial_y = topfarm_sgd_solve(aep_objective, init_x, init_y,
                                                  boundary, min_spacing, settings)
            trial_aep = get_aep(trial_x, trial_y)

            if trial_aep > best_aep_sofar:
                best_aep_sofar = trial_aep
                best_x_sofar = trial_x
                best_y_sofar = trial_y

            worst_stored_idx = jnp.argmin(stored_aep)
            if trial_aep > stored_aep[worst_stored_idx]:
                stored_x = stored_x.at[worst_stored_idx].set(trial_x)
                stored_y = stored_y.at[worst_stored_idx].set(trial_y)
                stored_aep = stored_aep.at[worst_stored_idx].set(trial_aep)

    # ── Consensus + refinement ────────────────────────────────────────
    temperature = 0.25
    shifted = stored_aep - jnp.max(stored_aep)
    softmax_w = jnp.exp(shifted / temperature)
    softmax_w = softmax_w / jnp.sum(softmax_w)

    ensemble_x = jnp.sum(stored_x * softmax_w[:, None], axis=0)
    ensemble_y = jnp.sum(stored_y * softmax_w[:, None], axis=0)

    refine = SGDSettings(
        learning_rate=25.0,
        max_iter=1000,
        additional_constant_lr_iterations=500,
        tol=1e-6,
        beta1=0.9,
        beta2=0.999,
        gamma_min_factor=0.01,
        ks_rho=100.0,
        spacing_weight=0.4,
        boundary_weight=0.4,
    )
    ref_x, ref_y = topfarm_sgd_solve(aep_objective, ensemble_x, ensemble_y,
                                      boundary, min_spacing, refine)
    ref_aep = get_aep(ref_x, ref_y)

    if ref_aep > best_aep_sofar:
        return ref_x, ref_y
    else:
        return best_x_sofar, best_y_sofar