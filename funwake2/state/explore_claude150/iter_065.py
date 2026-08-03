import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the exploration
# phase is re-cut as PULSE-WIDTH-MODULATED HEAT — the WSD/hold-then-decay bet
# from the prior-art menu (§6 "hold near c*D, then linear cool-down" / §2
# one-cycle) fused with the parent's proven restoration machinery. The parent
# spends half of every cosine cycle at middling lr that neither hops basins
# nor repairs constraints; here lr HOLDS at a hot plateau for ~80% of each
# cycle and drops only in brief, sharp REPAIR DIPS. Same three-cycle cadence,
# same anti-phased restoration, but the duty cycle is inverted: time above
# 1.3*D roughly doubles — the "longer, hotter exploration" the search state
# asks for — without touching the 5/5-seed-feasible endgame.
#
#   lr    — 3% warmup -> hot HOLD with a decaying envelope (1.5*D -> 1.05*D)
#           interrupted by three narrow dips to 0.65*D (dip = (1-cyc)^4, so
#           the plateau, not the trough, dominates each cycle), then the
#           proven straight linear tail landing exactly on gamma_min.
#   alpha — exploration floor 0.4*alpha0 during the hold; growing anti-phased
#           restoration bursts (4 -> 10 alpha0, stronger than the parent's
#           3 -> 8 because the hotter hold accrues more violation debt per
#           cycle) with a WIDER profile than the lr dip ((1-cyc)^3 vs ^4), so
#           repair brackets each dip — alpha is already rising as lr falls and
#           still elevated as lr recovers. After 62% the proven logistic ramp
#           to the bounded 6*alpha0 ALM plateau, then the proven cubic-delayed
#           geometric climb from 78% onto the terminal 5*alpha0*D/gamma_min.
#   betas — proven transitions (beta2 0.2 -> 0.9 at cool-down; beta1 gated
#           0.1 -> 0.02 in the terminal spike) plus the proven per-burst beta1
#           dip to 0.05 so momentum never drags turbines back across the
#           boundary mid-repair.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three hold/repair cycles inside the exploration phase
_HI0 = 1.50           # first hold level — HELD, not merely touched
_HI1 = 1.05           # final hold level; the linear tail starts here (proven)
_LO = 0.65            # repair-dip lr, in units of D (proven trough value)
_Q_LR = 4.0           # sharp lr dips: plateau occupies ~80% of each cycle
_Q_A = 3.0            # alpha bursts slightly wider than lr dips (coverage)
_A_LO = 0.4           # exploration penalty floor during the hold (proven)
_A_B0 = 4.0           # first restoration burst height, in alpha0 units
_A_B1 = 10.0          # last restoration burst height, in alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_BURST = 0.05      # reduced momentum inside each restoration burst
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> pulse-width-modulated hot hold -> linear tail ---
    # fc freezes at 1 past _F_COOL, where cyc = 1 and dip = 0, so the
    # cool-down starts exactly from the final hold level, _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    cyc = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * _N_CYC * fc))   # 1 on hold, 0 at dip centers
    dip = (1.0 - cyc) ** _Q_LR               # ~0 most of each cycle; 1 only mid-dip
    hi = _HI0 + (_HI1 - _HI0) * fc           # decaying hold envelope
    lr_cyc = (hi - (hi - _LO) * dip) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + growing anti-phased bursts -> plateau -> terminal climb ---
    # burst is wider than dip (Q_A < Q_LR): repair starts as lr falls and
    # persists as lr recovers; burst = 0 for frac >= _F_COOL (fc frozen on hold).
    burst = (1.0 - cyc) ** _Q_A
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * burst
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-burst beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_BURST) * burst            # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2