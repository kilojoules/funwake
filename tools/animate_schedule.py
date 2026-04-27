"""Capture turbine trajectories during a schedule_fn run + render MP4.

Mirrors playground/skeleton.py but exposes per-step (x, y) snapshots.

Usage:
    pixi run python tools/animate_schedule.py \\
        --schedule results_agent_schedule_only_5hr/iter_192.py \\
        --problem playground/problem.json \\
        --out paper/short_codex_figs/claude_animation.mp4 \\
        --frames 200 --fps 30
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon as MplPoly

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "playground", "pixwake", "src"))

from pixwake.optim.sgd import boundary_penalty, spacing_penalty
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit


def load_schedule_fn(path):
    spec = importlib.util.spec_from_file_location("user_sched", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.schedule_fn


def build_sim(info):
    D = info["rotor_diameter"]
    hub_height = info.get("hub_height", 150.0)
    t = info["turbine"]
    ws_arr = jnp.array(t["power_curve_ws"], dtype=float)
    power = jnp.array(t["power_curve_kw"], dtype=float)
    ct_ws = jnp.array(t.get("ct_curve_ws", t["power_curve_ws"]), dtype=float)
    ct = jnp.array(t["ct_curve_ct"], dtype=float)
    turbine = Turbine(
        rotor_diameter=D, hub_height=hub_height,
        power_curve=Curve(ws=ws_arr, values=power),
        ct_curve=Curve(ws=ct_ws, values=ct))
    sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))
    rose = info["wind_rose"]
    wd = jnp.array(rose["directions_deg"])
    ws = jnp.array(rose["speeds_ms"])
    weights = jnp.array(rose["weights"])
    return sim, wd, ws, weights


def init_layout(sim, n_target, boundary, min_spacing, wd, ws, weights, seed=0):
    """Wind-aware grid init, copy of skeleton.py logic."""
    def aep_objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6
    grad_obj = jax.grad(aep_objective, argnums=(0, 1))

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
        jnp.linspace(rx_min + min_spacing/2, rx_max - min_spacing/2, nx),
        jnp.linspace(ry_min + min_spacing/2, ry_max - min_spacing/2, ny))
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
        x, y = ix[indices], iy[indices]
    else:
        key = jax.random.PRNGKey(seed)
        x = jax.random.uniform(key, (n_target,), minval=float(x_min), maxval=float(x_max))
        key, _ = jax.random.split(key)
        y = jax.random.uniform(key, (n_target,), minval=float(y_min), maxval=float(y_max))
    gox, goy = grad_obj(x, y)
    lr0 = 50.0
    alpha0 = jnp.mean(jnp.abs(jnp.concatenate([gox, goy]))) / lr0
    return x, y, lr0, alpha0, grad_obj


def run_with_traj(schedule_fn, sim, n_target, boundary, min_spacing,
                  wd, ws, weights, total_steps=8000, frame_every=40, seed=0):
    """Run skeleton, record (x, y) every frame_every steps."""
    def con_penalty(x, y):
        return (boundary_penalty(x, y, boundary)
                + spacing_penalty(x, y, min_spacing))
    grad_con = jax.grad(con_penalty, argnums=(0, 1))

    x, y, lr0, alpha0, grad_obj = init_layout(
        sim, n_target, boundary, min_spacing, wd, ws, weights, seed=seed)

    @jax.jit
    def step(i, x, y, mx, my, vx, vy):
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
        eps = 1e-12
        x_new = x - lr * mx_hat / (jnp.sqrt(vx_hat) + eps)
        y_new = y - lr * my_hat / (jnp.sqrt(vy_hat) + eps)
        return x_new, y_new, mx_new, my_new, vx_new, vy_new

    mx = jnp.zeros_like(x); my = jnp.zeros_like(y)
    vx = jnp.zeros_like(x); vy = jnp.zeros_like(y)

    frames = [(np.array(x), np.array(y), 0)]
    for i in range(total_steps):
        x, y, mx, my, vx, vy = step(jnp.array(i), x, y, mx, my, vx, vy)
        if (i + 1) % frame_every == 0 or i + 1 == total_steps:
            frames.append((np.array(x), np.array(y), i + 1))
    return frames, lr0, alpha0


def render_animation(frames, boundary, wd, weights, out_path,
                     title="", rotor_radius=120.0, fps=30):
    fig, (ax_lay, ax_rose) = plt.subplots(
        1, 2, figsize=(12, 6),
        gridspec_kw={"width_ratios": [3, 1]})

    # Layout panel
    bnd = MplPoly(np.asarray(boundary), closed=True, fill=False,
                  edgecolor="0.2", lw=1.5)
    ax_lay.add_patch(bnd)
    bx_min, by_min = np.min(boundary, axis=0)
    bx_max, by_max = np.max(boundary, axis=0)
    pad = 0.05 * max(bx_max - bx_min, by_max - by_min)
    ax_lay.set_xlim(bx_min - pad, bx_max + pad)
    ax_lay.set_ylim(by_min - pad, by_max + pad)
    ax_lay.set_aspect("equal")
    ax_lay.set_xlabel("x [m]"); ax_lay.set_ylabel("y [m]")
    ax_lay.set_title(title, fontsize=11)
    scat = ax_lay.scatter([], [], s=140, c="C0", edgecolor="black",
                          linewidth=0.6, alpha=0.8)
    label = ax_lay.text(0.02, 0.97, "", transform=ax_lay.transAxes,
                        va="top", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    # Wind rose panel
    wd_arr = np.asarray(wd); w_arr = np.asarray(weights)
    n_dirs = len(wd_arr)
    ax_rose.remove()
    ax_rose = fig.add_subplot(1, 2, 2, projection="polar")
    ax_rose.set_theta_zero_location("N")
    ax_rose.set_theta_direction(-1)
    width = 2 * np.pi / max(n_dirs, 1)
    theta = np.deg2rad(wd_arr)
    if w_arr.ndim > 1:
        radial = w_arr.sum(axis=1) if w_arr.shape[0] == n_dirs else w_arr.sum(axis=0)
    else:
        radial = w_arr
    if radial.size != n_dirs:
        radial = np.ones(n_dirs) / n_dirs
    ax_rose.bar(theta, radial, width=width, bottom=0.0,
                color="C1", edgecolor="0.3", alpha=0.7)
    ax_rose.set_yticklabels([])
    ax_rose.set_title("Wind rose", fontsize=10)

    def update(frame_idx):
        x, y, step_idx = frames[frame_idx]
        scat.set_offsets(np.column_stack([x, y]))
        label.set_text(f"step {step_idx} / {frames[-1][2]}\n"
                       f"N = {len(x)} turbines")
        return scat, label

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000.0 / max(fps, 1), blit=False)

    print(f"[anim] writing {out_path} ...")
    anim.save(out_path, writer="ffmpeg", fps=fps, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schedule", required=True)
    p.add_argument("--problem", default="playground/problem.json")
    p.add_argument("--out", required=True)
    p.add_argument("--total-steps", type=int, default=8000)
    p.add_argument("--frame-every", type=int, default=40)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    problem = json.load(open(args.problem))
    sim, wd, ws, weights = build_sim(problem)
    boundary = jnp.array(problem["boundary_vertices"])
    n_target = int(problem["n_target"])
    min_spacing = float(problem["min_spacing_m"])

    schedule_fn = load_schedule_fn(args.schedule)
    title = args.title or os.path.basename(args.schedule)

    print(f"[anim] running {args.schedule} on {args.problem} ...")
    frames, lr0, alpha0 = run_with_traj(
        schedule_fn, sim, n_target, boundary, min_spacing, wd, ws, weights,
        total_steps=args.total_steps, frame_every=args.frame_every, seed=args.seed)
    print(f"[anim] {len(frames)} frames captured (lr0={lr0:.2f}, alpha0={alpha0:.4g})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    render_animation(frames, boundary, wd, weights, args.out,
                     title=title, fps=args.fps)
    print(f"[anim] done: {args.out}")


if __name__ == "__main__":
    main()
