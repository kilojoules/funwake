"""Particle swarm optimization with SGD local search.

Strategy:
1. Initialize population of layouts (particles)
2. Run simplified PSO for global exploration
3. Track global best and personal bests
4. After PSO, refine global best with SGD
5. Wind-aware population initialization
"""
import jax
import jax.numpy as jnp
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve, boundary_penalty, spacing_penalty


def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """PSO + SGD hybrid optimizer."""

    # ── Objective and constraints ──
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    def penalized_objective(x, y, alpha=100.0):
        aep = aep_objective(x, y)
        penalty = boundary_penalty(x, y, boundary) + spacing_penalty(x, y, min_spacing)
        return aep + alpha * penalty

    # ── Helper: create diverse initialization ──
    def create_particle(seed):
        x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
        x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

        # Mix of grid and random
        if seed == 0:
            # Hexagonal grid
            dx = min_spacing * 1.1
            dy = min_spacing * 1.1 * jnp.sqrt(3) / 2
        else:
            # Slightly perturbed grid
            factor = 1.0 + 0.1 * (seed / 5.0)
            dx = min_spacing * factor
            dy = min_spacing * factor * jnp.sqrt(3) / 2

        nx = max(3, int(jnp.ceil((x_max - x_min) / dx)) + 1)
        ny = max(3, int(jnp.ceil((y_max - y_min) / dy)) + 1)

        pts = []
        for i in range(ny):
            for j in range(nx):
                offset = dx / 2 if i % 2 == 1 else 0
                xi = x_min + j * dx + offset
                yi = y_min + i * dy
                pts.append([xi, yi])

        pts = jnp.array(pts)

        # Filter inside
        n_verts = boundary.shape[0]
        def is_inside(pt):
            def edge_dist(i):
                x1, y1 = boundary[i]
                x2, y2 = boundary[(i + 1) % n_verts]
                ex, ey = x2 - x1, y2 - y1
                el = jnp.sqrt(ex**2 + ey**2) + 1e-10
                return (pt[0] - x1) * (-ey / el) + (pt[1] - y1) * (ex / el)
            return jnp.min(jax.vmap(edge_dist)(jnp.arange(n_verts))) > min_spacing * 0.3

        inside_mask = jax.vmap(is_inside)(pts)
        inside_pts = pts[inside_mask]

        if len(inside_pts) >= n_target:
            key = jax.random.PRNGKey(seed * 100)
            indices = jax.random.choice(key, len(inside_pts), (n_target,), replace=False)
            x, y = inside_pts[indices, 0], inside_pts[indices, 1]
        else:
            key = jax.random.PRNGKey(seed * 100)
            x = jax.random.uniform(key, (n_target,), minval=x_min, maxval=x_max)
            key, _ = jax.random.split(key)
            y = jax.random.uniform(key, (n_target,), minval=y_min, maxval=y_max)

        return x, y

    # ── Initialize swarm ──
    n_particles = 6
    particles_x = []
    particles_y = []
    velocities_x = []
    velocities_y = []
    personal_best_x = []
    personal_best_y = []
    personal_best_score = []

    for i in range(n_particles):
        x, y = create_particle(i)
        particles_x.append(x)
        particles_y.append(y)
        velocities_x.append(jnp.zeros_like(x))
        velocities_y.append(jnp.zeros_like(y))

        score = penalized_objective(x, y)
        personal_best_x.append(x)
        personal_best_y.append(y)
        personal_best_score.append(score)

    # Find initial global best
    global_best_idx = jnp.argmin(jnp.array(personal_best_score))
    global_best_x = personal_best_x[int(global_best_idx)]
    global_best_y = personal_best_y[int(global_best_idx)]
    global_best_score = personal_best_score[int(global_best_idx)]

    # ── PSO loop (simplified, limited iterations) ──
    w = 0.7  # Inertia
    c1 = 1.5  # Cognitive
    c2 = 1.5  # Social
    n_iterations = 50

    key = jax.random.PRNGKey(42)

    for iteration in range(n_iterations):
        for i in range(n_particles):
            # Update velocity
            key, k1, k2 = jax.random.split(key, 3)
            r1 = jax.random.uniform(k1, shape=particles_x[i].shape)
            r2 = jax.random.uniform(k2, shape=particles_x[i].shape)

            velocities_x[i] = (w * velocities_x[i] +
                               c1 * r1 * (personal_best_x[i] - particles_x[i]) +
                               c2 * r2 * (global_best_x - particles_x[i]))
            velocities_y[i] = (w * velocities_y[i] +
                               c1 * r1 * (personal_best_y[i] - particles_y[i]) +
                               c2 * r2 * (global_best_y - particles_y[i]))

            # Limit velocity
            v_max = min_spacing * 0.5
            velocities_x[i] = jnp.clip(velocities_x[i], -v_max, v_max)
            velocities_y[i] = jnp.clip(velocities_y[i], -v_max, v_max)

            # Update position
            x_min, y_min = float(jnp.min(boundary[:, 0])), float(jnp.min(boundary[:, 1]))
            x_max, y_max = float(jnp.max(boundary[:, 0])), float(jnp.max(boundary[:, 1]))

            particles_x[i] = jnp.clip(particles_x[i] + velocities_x[i], x_min, x_max)
            particles_y[i] = jnp.clip(particles_y[i] + velocities_y[i], y_min, y_max)

            # Evaluate
            score = penalized_objective(particles_x[i], particles_y[i])

            # Update personal best
            if score < personal_best_score[i]:
                personal_best_score[i] = score
                personal_best_x[i] = particles_x[i]
                personal_best_y[i] = particles_y[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_x = particles_x[i]
                    global_best_y = particles_y[i]

    # ── Refine global best with SGD ──
    settings = SGDSettings(
        learning_rate=100.0,
        max_iter=3000,
        additional_constant_lr_iterations=1500,
        tol=1e-6,
        beta1=0.1,
        beta2=0.2,
        gamma_min_factor=0.005,
        ks_rho=100.0,
        spacing_weight=80.0,
        boundary_weight=80.0,
    )

    opt_x, opt_y = topfarm_sgd_solve(
        aep_objective, global_best_x, global_best_y,
        boundary, min_spacing, settings)

    return opt_x, opt_y
