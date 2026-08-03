import jax.numpy as jnp

# STRUCTURALLY NEW vs the WSD-trapezoid best (+0.0397%): the flat hot hold is
# replaced by the two untried menu directions the search state calls for,
# composed so each covers the other's weakness:
#   lr    — MULTI-CYCLE COSINE (untried): after a 3% warmup, TWO full cosine
#           cycles swing lr between 1.30*D and 0.62*D over the first 62% of
#           the run. Peaks are HOTTER than the best's 1.12*D hold (the "push
#           the peak higher" bet) and the troughs stay near 0.6*D — cyclic
#           exploration, NOT the SGDR failure mode of dropping to the floor
#           mid-run. From 62% a straight linear cool-down lands lr exactly on
#           gamma_min at the last step (the proven WSD tail is kept).
#   alpha — MID-RUN FEASIBILITY-RESTORATION BURSTS (untried) on top of the
#           proven epsilon-contraction. Baseline penalty stays at the proven
#           0.5*alpha0 for free exploration, but a von-Mises bump locked to
#           the lr troughs (peak 25*alpha0, ~270-step width) repairs boundary/
#           spacing debt exactly when lr is lowest — hot peaks create damage,
#           cool troughs pay it off — so the hotter peaks cannot bank
#           unrepairable violations. The terminal blow-up is IDENTICAL to the
#           best's: geometric growth from 0.5*alpha0 to 5*alpha0*D/gamma_min
#           (cubic-delayed from 55%), preserving strict 5/5 feasibility.
#   betas — proven transitions kept: beta2 0.2 -> 0.9 at the cool-down start,
#           beta1 0.1 -> 0.02 in the terminal phase; NEW: beta1 also dips to
#           0.03 inside each repair burst so momentum cannot drag turbines
#           back across the boundary mid-repair (menu bet 4 applied to the
#           bursts, not just the endgame).
_C_HI = 1.30           # cycle peak lr (hotter than the best's 1.12*D hold)
_C_LO = 0.62           # cycle trough lr — still hot, never near the floor
_N_CYC = 2.0           # two full cosine cycles inside the exploration phase
_F_WARM = 0.03         # linear warmup over first 3% so the grid init survives
_F_COOL = 0.62         # exploration ends here; linear decay to gamma_min at 100%
_KAPPA = 12.0          # burst sharpness: bumps peaked exactly at the lr troughs
_ALPHA_LO = 0.5        # exploration penalty (proven), in alpha0 units
_BURST_GAIN = 25.0     # burst peak alpha = 25*alpha0 (repair, not blow-up)
_TERM_GAIN = 5.0       # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_F_CONTRACT = 0.55     # terminal tolerance contraction starts here
_POW = 3.0             # cubic back-loading of the contraction
_BETA2_LO = 0.2
_BETA2_HI = 0.9
_B2_CENTER = 0.62      # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1           # native momentum while exploring
_B1_BURST = 0.03       # low momentum inside each mid-run repair burst
_B1_LO = 0.02          # near-zero momentum during the terminal alpha blow-up
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> two cosine cycles in [1.30, 0.62]*D -> linear tail ---
    # Clipping frac freezes the phase at a peak (cos=1) for frac > _F_COOL, so
    # the cool-down starts from the hot value and the burst term shuts off.
    phase = 2.0 * jnp.pi * _N_CYC * jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    lr_cyc = (_C_LO + (_C_HI - _C_LO) * 0.5 * (1.0 + jnp.cos(phase))) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)
    warm = jnp.minimum(frac / _F_WARM, 1.0)
    lr = lr_env * warm

    # --- alpha: repair bursts at the lr troughs + terminal contraction ---
    # burst = 1 exactly where cos(phase) = -1 (the lr troughs at 15.5% and
    # 46.5%), ~exp(-2*kappa) elsewhere; both spikes and the endgame divergence
    # act multiplicatively on the proven 0.5*alpha0 exploration floor.
    burst = jnp.exp(-_KAPPA * (1.0 + jnp.cos(phase)))
    s = jnp.clip((frac - _F_CONTRACT) / (1.0 - _F_CONTRACT), 0.0, 1.0) ** _POW
    log_term = jnp.log((_TERM_GAIN / _ALPHA_LO) * Dj / gmin)
    log_burst = jnp.log(_BURST_GAIN / _ALPHA_LO)
    alpha = _ALPHA_LO * alpha0 * jnp.exp(s * log_term + burst * log_burst)

    # --- betas: beta2 up for the polish phase; beta1 down in bursts + endgame ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _BETA2_LO + (_BETA2_HI - _BETA2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    b1_expl = _B1_HI - (_B1_HI - _B1_BURST) * jnp.clip(burst, 0.0, 1.0)
    beta1 = b1_expl + (_B1_LO - b1_expl) * b1r

    return lr, alpha, beta1, beta2