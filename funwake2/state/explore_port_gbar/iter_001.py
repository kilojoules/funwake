"""Schedule: WSD learning rate (warmup → plateau at c·D → log-linear cool-down
to gamma_min → polish floor) with a delayed, spiked penalty ramp on top of the
native alpha ∝ 1/lr backbone, and Adam moments phase-transitioned with lr.

Rationale vs. the parent (native pixwake coupling):
  * lr — native decays as lr0·exp(-mid·k²/2), i.e. it lingers at the exploration
    scale and then collapses, giving only a few hundred steps below ~1 m. Here
    the cool-down is log-linear, so every *spatial decade* between c·D and
    gamma_min gets an equal share of steps → far more fine-scale refinement.
  * alpha — keeps the native 1/lr backbone (so terminal feasibility is still
    guaranteed by lr → gamma_min) but multiplies it by a graduated-penalty
    factor: ~0.45× while exploring (homotopy / graduated constraint), → 1× by
    mid-run, then a terminal ×3 feasibility-restoration spike. The spike shrinks
    the equilibrium violation to ~gamma_min/6 instead of ~gamma_min/2, buying
    the margin that pays for the looser exploration phase.
  * betas — 0.1/0.2 (≈ sign descent, step ≈ lr metres) during exploration, then
    ramped anti-correlated with lr to 0.8/0.9 so the small-lr polish phase
    descends on an averaged, better-conditioned direction.
"""

import jax.numpy as jnp

_C = 200.0 / 240.0        # exploration lr per rotor diameter (DEI-tuned)

_WARM = 0.02              # warmup fraction of the run
_HOLD = 0.40              # end of the stable exploration plateau
_POLISH = 0.96            # lr reaches gamma_min at this point of the cool-down

_M_LO = 0.45              # penalty multiplier while exploring
_RAMP0, _RAMP1 = 0.35, 0.72   # graduated ramp window (fractions of the run)
_SPIKE0 = 0.90            # terminal feasibility-restoration spike starts
_M_HI = 3.0               # terminal penalty multiplier

_B1_LO, _B1_HI = 0.1, 0.80
_B2_LO, _B2_HI = 0.2, 0.90
_BETA_END = 0.90          # betas finish ramping here


def _f(x):
    """Cast to a float jnp scalar without float()/int() (step/alpha0 are traced)."""
    return jnp.asarray(x) * 1.0


def _smoothstep(x):
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    Dm = _f(D)
    T = jnp.maximum(_f(total_steps) - 1.0, 1.0)
    t = jnp.clip(_f(step) / T, 0.0, 1.0)              # run progress in [0, 1]

    lr0 = _C * Dm                                     # exploration scale from D
    gm = jnp.clip(_f(gamma_min), 1e-8, 0.5 * lr0)     # terminal tolerance (m)

    # --- learning rate: warmup -> plateau -> log-linear cool-down -> floor ----
    warm = 0.25 + 0.75 * _smoothstep(t / _WARM)
    q = jnp.clip((t - _HOLD) / (1.0 - _HOLD), 0.0, 1.0)   # cool-down progress
    g = jnp.clip(q / _POLISH, 0.0, 1.0)                   # 1.0 during the polish tail
    lr = jnp.maximum(lr0 * warm * (gm / lr0) ** g, gm)

    # --- penalty: native alpha0*D/lr backbone, graduated + terminal spike -----
    ramp = _M_LO + (1.0 - _M_LO) * _smoothstep((t - _RAMP0) / (_RAMP1 - _RAMP0))
    spike = 1.0 + (_M_HI - 1.0) * _smoothstep((t - _SPIKE0) / (1.0 - _SPIKE0))
    alpha = alpha0 * Dm / jnp.maximum(lr, 1e-30) * ramp * spike

    # --- Adam moments: sign-like while exploring, averaged while polishing ----
    r = _smoothstep((t - _HOLD) / (_BETA_END - _HOLD))
    beta1 = _B1_LO + (_B1_HI - _B1_LO) * r
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * r

    return lr, alpha, beta1, beta2