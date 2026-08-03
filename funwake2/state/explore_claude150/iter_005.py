import jax.numpy as jnp

# STRUCTURALLY NEW vs the one-cycle parent: SGDR-style COSINE WARM RESTARTS with
# MID-RUN FEASIBILITY-RESTORATION BURSTS (both untried directions from the menu).
#   lr    — exploration phase (first 70%) is 3 cosine cycles with geometrically
#           decaying peaks (1.2*D -> 0.74*D -> 0.46*D): a HOTTER first peak than
#           the parent, re-injected twice by restarts instead of one long hold.
#           Each restart has a short intra-cycle warmup so the just-repaired
#           layout survives the lr jump. Final 30%: half-cosine cool-down from
#           the cycle floor landing exactly on gamma_min.
#   alpha — CYCLIC, anti-correlated with lr inside each cycle: low (0.5*alpha0)
#           while lr is high (free exploration), logistic burst to ~6*alpha0 as
#           each cycle cools (a feasibility-restoration burst that repairs
#           violations before the next restart). After the last cycle the burst
#           stays latched, becoming the parent's proven ~6*alpha0 plateau; the
#           parent's PROVEN TERMINAL DIVERGENCE (gain 5, gate at 90%, alpha ->
#           5*alpha0*D/gamma_min) is preserved unchanged for strict feasibility.
#   beta2 — native 0.2 while cycling, ramped to 0.9 at the final cool-down
#           (parent-proven phase transition); beta1 held at native 0.1.
_F_EXPL = 0.70         # exploration (restart) phase ends at 70% of the run
_N_CYCLES = 3.0        # three cosine warm-restart cycles
_PEAK0 = 1.2           # first-cycle lr peak = 1.2 * D (hotter than parent's 1.0)
_PEAK_DECAY = 0.62     # geometric peak decay per restart
_LR_FLOOR = 0.15       # inter-cycle lr floor = 0.15 * D (repairs happen here)
_CYC_WARM = 0.06       # intra-cycle linear warmup over first 6% of each cycle
_ALPHA_LO = 0.5        # exploration penalty (parent-proven), in alpha0 units
_ALPHA_HI = 6.0        # burst top = parent's proven plateau, in alpha0 units
_BURST_CENTER = 0.75   # burst engages over the last quarter of each cycle
_BURST_WIDTH = 0.06
_GATE_CENTER = 0.90    # terminal restoration: identical to feasible parent
_GATE_WIDTH = 0.02
_TERM_GAIN = 5.0       # terminal alpha ~ 5*alpha0*D/lr_env (parent-proven scale)
_BETA2_LO = 0.2
_BETA2_HI = 0.9
_B2_CENTER = 0.70      # beta2 transition aligned with final cool-down start
_B2_WIDTH = 0.05


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    gmin = jnp.asarray(gamma_min) * 1.0
    lr_floor = _LR_FLOOR * D

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr, exploration: cosine warm restarts with decaying peaks ---
    u = jnp.clip(frac / _F_EXPL, 0.0, 1.0)
    k = jnp.clip(jnp.floor(u * _N_CYCLES), 0.0, _N_CYCLES - 1.0)
    cycle_pos = jnp.clip(u * _N_CYCLES - k, 0.0, 1.0)   # pins at 1 after 70%
    peak = _PEAK0 * D * _PEAK_DECAY ** k
    lr_exp = lr_floor + 0.5 * (peak - lr_floor) * (1.0 + jnp.cos(jnp.pi * cycle_pos))
    # short warmup after every restart so the just-repaired layout isn't blown up
    lr_exp = lr_exp * jnp.minimum(cycle_pos / _CYC_WARM, 1.0)

    # --- lr, final 30%: half-cosine from the cycle floor down to gamma_min ---
    p = jnp.clip((frac - _F_EXPL) / (1.0 - _F_EXPL), 0.0, 1.0)
    lr_fin = gmin + 0.5 * (lr_floor - gmin) * (1.0 + jnp.cos(jnp.pi * p))
    lr = jnp.where(frac <= _F_EXPL, lr_exp, lr_fin)     # continuous at the seam

    # --- alpha: per-cycle restoration bursts, anti-correlated with lr ---
    burst = 1.0 / (1.0 + jnp.exp(-(cycle_pos - _BURST_CENTER) / _BURST_WIDTH))
    alpha_cyc = alpha0 * (_ALPHA_LO + (_ALPHA_HI - _ALPHA_LO) * burst)
    # (after 70%, cycle_pos is pinned at 1 -> burst latches into the ~6*alpha0
    #  plateau the feasible parent used through its cool-down)

    # --- terminal feasibility restoration: parent-proven gated divergence ---
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _GATE_CENTER) / _GATE_WIDTH))
    alpha_term = _TERM_GAIN * alpha0 * D / jnp.maximum(lr_fin, 1e-30)
    alpha = alpha_cyc + gate * alpha_term

    # --- betas: native 0.1/0.2 while cycling, beta2 up for the polish phase ---
    beta1 = 0.1
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _BETA2_LO + (_BETA2_HI - _BETA2_LO) * b2r

    return lr, alpha, beta1, beta2