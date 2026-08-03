import jax.numpy as jnp

# STRUCTURAL DEPARTURE from the native 1/lr coupling (the parent's core):
# 1. lr: SGDR — three cosine cycles with warm restarts and geometrically
#    decaying peaks (hot exploration waves instead of one plateau+decay).
#    The final cycle's cosine floor is gamma_min, so lr still lands there.
# 2. alpha: DECOUPLED during exploration — a bounded ~0.55*alpha0 penalty
#    (delayed/weakened ramp, exact-penalty guidance) so hot cycles can cross
#    constraint boundaries to reach better basins, plus mid-run FEASIBILITY-
#    RESTORATION BURSTS (Gaussian alpha bumps) at each cycle trough, where lr
#    is low, to pull violators back before the next restart.
# 3. Terminal ~10%: smooth logistic handover BACK to the native coupling
#    alpha0*D/lr with the parent's proven logistic spike — the endgame that
#    delivered 5/5 feasibility is preserved exactly in spirit.
# 4. betas: one-cycle-style anti-correlation with lr (low momentum at hot
#    peaks, higher momentum + longer beta2 memory in the troughs), blending
#    to native (0.1, 0.2) in the terminal feasibility phase.

_C = 230.0 / 240.0                 # hot-peak diameter rule (kept from parent)
_WARM_DEN = 50                     # linear lr warmup over the first ~2% of steps

_B1, _B2 = 0.32, 0.62              # warm-restart boundaries (run fractions)
_P1, _P2, _P3 = 1.00, 0.62, 0.40   # decaying cycle peaks (x lr0)
_F1, _F2 = 0.10, 0.06              # cycle floors (x lr0); cycle-3 floor = gamma_min

_ALPHA_EXPLORE = 0.55              # bounded exploration penalty (x alpha0)
_BURST_AMP = 12.0                  # restoration-burst height (x alpha0)
_BURST_C1, _BURST_C2 = 0.305, 0.605
_BURST_W = 0.012                   # burst width (run fraction)

_BLEND_CENTER = 0.90               # terminal handover to native coupling
_BLEND_W = 0.02
_SPIKE_GAIN = 4.0                  # parent's terminal alpha spike, unchanged
_SPIKE_CENTER = 0.965
_SPIKE_W = 0.012


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    Df = float(D)
    lr0 = _C * Df                            # exploration lr scale from D
    g = float(gamma_min)
    n_total = float(total_steps)
    n_warm = max(int(total_steps) // _WARM_DEN, 1)

    def sig(x):
        return 1.0 / (1.0 + jnp.exp(-x))

    f = (step + 1.0) / n_total               # run fraction, traceable

    # --- lr: three cosine half-waves with instant warm restarts ------------
    p1 = jnp.clip(f / _B1, 0.0, 1.0)
    p2 = jnp.clip((f - _B1) / (_B2 - _B1), 0.0, 1.0)
    p3 = jnp.clip((f - _B2) / (1.0 - _B2), 0.0, 1.0)
    lr1 = _F1 * lr0 + (_P1 - _F1) * lr0 * 0.5 * (1.0 + jnp.cos(jnp.pi * p1))
    lr2 = _F2 * lr0 + (_P2 - _F2) * lr0 * 0.5 * (1.0 + jnp.cos(jnp.pi * p2))
    lr3 = g + (_P3 * lr0 - g) * 0.5 * (1.0 + jnp.cos(jnp.pi * p3))
    lr_env = jnp.where(f < _B1, lr1, jnp.where(f < _B2, lr2, lr3))
    # cycle 3 ends exactly at gamma_min: cos(pi)=-1 -> lr = g.

    # Short linear warmup so the feasible grid init survives the first peak;
    # scales lr only, never alpha.
    warm = jnp.minimum((step + 1.0) / n_warm, 1.0)
    lr = lr_env * warm

    # --- alpha: bounded + bursts, then terminal native coupling ------------
    burst = _BURST_AMP * (
        jnp.exp(-0.5 * ((f - _BURST_C1) / _BURST_W) ** 2)
        + jnp.exp(-0.5 * ((f - _BURST_C2) / _BURST_W) ** 2))
    alpha_explore = alpha0 * (_ALPHA_EXPLORE + burst)

    # Native coupling on the cyclic envelope: alpha0 * D / lr = mean|grad J|/lr,
    # diverging as lr -> gamma_min — the same late-feasibility guarantee as the
    # parent — boosted by the parent's terminal logistic spike.
    alpha_native = alpha0 * Df / jnp.maximum(lr_env, 1e-30)
    spike = 1.0 + _SPIKE_GAIN * sig((f - _SPIKE_CENTER) / _SPIKE_W)

    s = sig((f - _BLEND_CENTER) / _BLEND_W)  # 0 during exploration, 1 at the end
    alpha = (1.0 - s) * alpha_explore + s * alpha_native * spike

    # --- betas: anti-correlated with lr, native (0.1, 0.2) in the endgame --
    rho = jnp.clip(lr_env / lr0, 0.0, 1.0)
    beta1 = (1.0 - s) * (0.1 + 0.6 * (1.0 - rho)) + s * 0.1
    beta2 = (1.0 - s) * (0.2 + 0.55 * (1.0 - rho)) + s * 0.2

    return lr, alpha, beta1, beta2