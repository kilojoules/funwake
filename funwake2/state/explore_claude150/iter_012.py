import jax.numpy as jnp

# STRUCTURALLY NEW vs the decaying-peak-restart best (+0.0450%): the two menu
# directions the search state lists as STILL untried — ANTI-PHASE CYCLIC ALPHA
# synced to the SGDR restarts, which IS a train of MID-RUN FEASIBILITY-
# RESTORATION BURSTS — composed on the proven skeleton (3% warmup, linear tail
# landing exactly on gamma_min, beta2 phase transition, terminal alpha spike).
#
#   lr    — three decaying-peak cosine restarts as in the best, but HOTTER
#           first peak (1.65*D vs 1.55*D, per the parent hint) and DEEPER
#           troughs (0.55*D vs 0.65*D). The deep troughs are not a weakness:
#           they are where the alpha bursts land, so small precise steps
#           carry turbines back inside the boundary. From 60% the proven WSD
#           linear tail runs from the final 1.05*D peak onto gamma_min.
#   alpha — BREATHES IN ANTI-PHASE WITH lr (never tied to 1/lr): a 0.3*alpha0
#           floor at the hot lr peaks frees basin-hopping to trade violation
#           for AEP, and a sharp cubic-shaped burst to 9*alpha0 at each lr
#           trough repays the violation debt IMMEDIATELY (filter/funnel-style
#           restoration) instead of deferring it all to the endgame. The
#           proven logistic ramp then lifts alpha onto the bounded 6*alpha0
#           ALM plateau for the polish, and the proven cubic-delayed
#           geometric climb from 78% ends at 5*alpha0*D/gamma_min — the
#           terminal restoration that made the parent line 5/5-seed feasible.
#   betas — proven transitions only: beta2 0.2 -> 0.9 at the cool-down start;
#           beta1 flat 0.1, gated to 0.02 during the terminal spike so the
#           diverging alpha never rides momentum.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.60        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three restarts inside the exploration phase
_HI0 = 1.65           # hottest first peak yet, in units of D
_HI1 = 1.05           # final peak — the linear tail starts from here (proven)
_LO = 0.55            # deep bounded troughs where the alpha bursts land
_A_FLOOR = 0.3        # penalty floor at the hot lr peaks, in alpha0 units
_A_BURST = 9.0        # restoration-burst peak at each lr trough
_Q = 3.0              # cubic burst shaping: narrow bursts, wide free phases
_A_PLAT = 6.0         # bounded ALM plateau through the polish (proven)
_A_CENTER = 0.62      # logistic ramp onto the plateau (proven constants)
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
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
    cyc = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * _N_CYC * fc))   # 1 at peaks, 0 at troughs
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying peak envelope
    lr_cyc = (_LO + (hi - _LO) * cyc) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: anti-phase bursts -> logistic ramp -> plateau -> terminal climb ---
    # (1 - cyc)^Q is ~0 across the hot peaks and spikes to 1 exactly at the lr
    # troughs: three mid-run feasibility-restoration bursts, then cyc freezes
    # at 1 so the exploration term parks on the floor before the ramp lifts it.
    expl = _A_FLOOR + (_A_BURST - _A_FLOOR) * (1.0 - cyc) ** _Q
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    base = expl + (_A_PLAT - expl) * ramp                     # in alpha0 units
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * base * jnp.exp(s * log_term)             # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2