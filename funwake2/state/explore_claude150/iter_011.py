import jax.numpy as jnp

# STRUCTURALLY NEW vs the multi-cycle-cosine best (+0.0428%): two menu
# directions the search state explicitly lists as UNTRIED, composed so each
# covers the other's weakness, on top of the proven skeleton (3% warmup,
# linear tail onto gamma_min, beta2 phase transition, terminal alpha spike).
#
#   lr    — SGDR-STYLE DECAYING-PEAK RESTARTS: three cosine cycles whose
#           peaks DECAY 1.55*D -> 1.05*D (the best used two FIXED 1.30*D
#           peaks). The first restart is far hotter than anything tried —
#           big basin-hopping kicks exactly when the layout is farthest from
#           a good basin — while later peaks cool so mid-run refinement is
#           not destroyed. Troughs are pinned at 0.65*D (bounded; never the
#           SGDR drop-to-the-floor failure mode). From 60% the proven WSD
#           linear tail lands lr exactly on gamma_min at the last step,
#           starting from the 1.05*D final peak (between the proven 1.12*D
#           and 1.30*D tail starts).
#   alpha — DECOUPLED LOGISTIC RAMP-THEN-PLATEAU (prior-art bets 1+2: bounded
#           ALM plateau + delayed ramp; NOT tied to 1/lr, NO trough bursts).
#           A 0.4*alpha0 floor frees the hot restarts to trade violation for
#           AEP; ONE logistic ramp centered at the cool-down start lifts
#           alpha to a bounded 6*alpha0 plateau, so hot-phase violation debt
#           is repaid CONTINUOUSLY through the polish while lr is still
#           effective (penalty gradients vanish once feasible, so the
#           plateau costs no AEP late). A cubic-delayed geometric climb from
#           78% carries alpha from the plateau to the PROVEN terminal
#           5*alpha0*D/gamma_min, preserving the 5/5-seed feasible endgame.
#   betas — proven transitions only: beta2 0.2 -> 0.9 at the cool-down
#           start; beta1 flat 0.1 exploring, gated down to 0.02 during the
#           terminal spike so diverging alpha never rides momentum.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.60        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three restarts inside the exploration phase
_HI0 = 1.55           # first restart peak (hottest lr the search has tried)
_HI1 = 1.05           # last restart peak — the linear tail starts from here
_LO = 0.65            # bounded trough lr, in units of D
_A_LO = 0.4           # exploration penalty floor, in alpha0 units
_A_PLAT = 6.0         # bounded ALM-style plateau, in alpha0 units
_A_CENTER = 0.62      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here
_POW = 3.0            # cubic back-loading of the terminal climb
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> 3 decaying-peak cosine restarts -> linear tail ---
    # fc freezes at 1 past _F_COOL, so cos(2*pi*N) = 1 pins the cool-down
    # start at the final (coolest) peak, _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    cyc = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * _N_CYC * fc))   # 1 at peaks
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying peak envelope
    lr_cyc = (_LO + (hi - _LO) * cyc) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor -> logistic ramp -> bounded plateau -> terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    plateau = _A_LO + (_A_PLAT - _A_LO) * ramp                # in alpha0 units
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * plateau * jnp.exp(s * log_term)          # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2