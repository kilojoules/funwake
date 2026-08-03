import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-notch best (+0.0577%): two coupled bets
# that change the OPTIMIZATION DYNAMICS, not just the lr waveform constants.
#
# 1) FRONT-LOADED HEAT PROTOCOL (two-block mesa). Every prior attempt spread
#    its heat quasi-uniformly across the exploration phase. Here the budget is
#    front-loaded: a sustained SUPER-HOT MESA at 1.85*D for the first ~26% —
#    hotter than any sustained level tried — while the layout is far from
#    converged and boundary debt is cheapest to repair, then a step-down to a
#    modest working hold (1.30 -> 1.05*D) until the proven 62% cool-down.
#    Only TWO repair notches (Gaussian, deep to 0.30*D): one at the mesa->hold
#    transition to settle the debt the mesa ran up, one mid-hold.
#
# 2) MOMENTUM ARCH (Sutskever ramp -> Demon decay) — the prior-art menu's
#    untried beta1 hypothesis (§4). All attempts so far kept beta1 pinned at
#    the native 0.1; in Adam, beta1 sets direction COHERENCE, not step size,
#    so it can be raised without blowing up the update norm. beta1 ramps
#    0.1 -> 0.8 across exploration (coherent collective turbine drift through
#    flat wake-interaction regions), Demon-decays back to ~0.1 across the
#    cool-down for responsive polishing, crashes to 0.03 inside each repair
#    notch (momentum must not carry turbines back over the boundary) and to
#    the proven 0.02 during the terminal alpha spike.
#
# The 5/5-seed-feasible endgame is preserved EXACTLY: linear lr tail landing
# on gamma_min at the last step, logistic ramp to the bounded 6*alpha0 ALM
# plateau, and the cubic-delayed geometric climb from 78% to the terminal
# 5*alpha0*D/gamma_min spike. The alpha floor is raised to 0.75*alpha0 during
# the mesa only, to counterbalance the extra heat.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_MESA_END = 0.26      # smooth step-down center: mesa -> working hold
_MESA_W = 0.02
_HI_MESA = 1.85       # sustained super-hot mesa level, in units of D
_HI2_0 = 1.30         # working-hold start level
_HI2_1 = 1.05         # working-hold end level; the linear tail launches here
_F2_0 = 0.28          # working-hold envelope decays from here to _F_COOL
_LO = 0.30            # notch-bottom lr — deep, surgical repair windows
_NC_A = 0.26          # repair notch A center (mesa transition)
_NC_B = 0.45          # repair notch B center (mid working hold)
_NW = 0.015           # notch Gaussian sigma (~120 of 8000 steps)
_A_LO = 0.4           # exploration penalty floor, in alpha0 units (proven)
_A_MESA_EXTRA = 0.35  # extra alpha floor during the mesa (0.75*alpha0 total)
_A_BA = 4.0           # restoration burst A height, in alpha0 units
_A_BB = 7.0           # restoration burst B height (debt grows -> stronger)
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp centered after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62     # beta2 transition aligned with cool-down start (proven)
_B2_WIDTH = 0.05
_B1_BASE = 0.1        # native momentum at the run's ends
_B1_PEAK = 0.8        # momentum-arch peak during hot exploration
_B1_UP_C = 0.18       # Sutskever ramp-up center
_B1_UP_W = 0.06
_B1_DN_C = 0.70       # Demon decay center (inside the cool-down)
_B1_DN_W = 0.05
_B1_NOTCH = 0.03      # momentum crash inside each repair notch
_B1_TERM = 0.02       # near-zero momentum during the terminal alpha spike
_B1_TC = 0.88
_B1_TW = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- repair notches: two localized Gaussians, ~0 by the cool-down start ---
    nA = jnp.exp(-0.5 * ((frac - _NC_A) / _NW) ** 2)
    nB = jnp.exp(-0.5 * ((frac - _NC_B) / _NW) ** 2)
    notch = jnp.clip(nA + nB, 0.0, 1.0)

    # --- lr: warmup -> super-hot mesa -> step-down working hold -> tail ---
    mix2 = 1.0 / (1.0 + jnp.exp(-(frac - _MESA_END) / _MESA_W))   # 0 in mesa, 1 after
    fc2 = jnp.clip((frac - _F2_0) / (_F_COOL - _F2_0), 0.0, 1.0)
    hi = _HI_MESA * (1.0 - mix2) + (_HI2_0 + (_HI2_1 - _HI2_0) * fc2) * mix2
    lr_hold = (_LO + (hi - _LO) * (1.0 - notch)) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)                  # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                       # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: mesa-raised floor + notch bursts -> plateau -> terminal climb ---
    floor = _A_LO + _A_MESA_EXTRA * (1.0 - mix2)                  # 0.75 in mesa, 0.4 after
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = floor + (_A_PLAT - _A_LO) * ramp + _A_BA * nA + _A_BB * nB
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)          # ends at 5*alpha0*D/gmin

    # --- beta2: proven 0.2 -> 0.9 transition at the cool-down ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    # --- beta1: momentum arch, notch crashes, terminal gate ---
    up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_C) / _B1_UP_W))
    dn = 1.0 / (1.0 + jnp.exp(-(frac - _B1_DN_C) / _B1_DN_W))
    b1_arch = _B1_BASE + (_B1_PEAK - _B1_BASE) * up * (1.0 - dn)
    b1 = b1_arch * (1.0 - notch) + _B1_NOTCH * notch              # crash while repairing
    tg = 1.0 / (1.0 + jnp.exp(-(frac - _B1_TC) / _B1_TW))
    beta1 = b1 + (_B1_TERM - b1) * tg

    return lr, alpha, beta1, beta2