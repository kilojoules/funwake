import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the lr waveform
# is rebuilt from the strongest UNTRIED lr idea in the prior-art menu (§6:
# hold near c*D, WSD-style, beats cosine) — FLAT-TOP HOT HOLDS WITH NARROW
# GAUSSIAN RESTORATION DIPS replacing the cosine restarts. A cosine cycle
# spends only ~1/3 of its steps near the peak; a flat-top wave spends ~3/4 of
# each cycle AT the hot hold, so the same exploration phase delivers ~15% more
# cumulative hot-lr basin hopping — exactly the "higher/LONGER lr peak early"
# the guidance asks for. The dip is a narrow mid-cycle Gaussian: lr plunges
# briefly to the trough while a sharpened alpha burst (strengthened vs parent,
# since it has fewer steps to work with) repairs the violation debt, then the
# hold resumes. The proven machinery that made the parent 5/5-seed feasible is
# preserved verbatim: anti-phased restoration bursts (here Gaussian, exactly
# coincident with the lr dips), the 62% cool-down with a straight linear tail
# landing on gamma_min, the logistic ramp to the bounded 6*alpha0 ALM plateau,
# the cubic-delayed geometric terminal climb to 5*alpha0*D/gamma_min, the
# beta2 0.2 -> 0.9 cool-down transition, the per-burst beta1 dip, and the
# terminal beta1 -> 0.02 gate.
#
#   lr    — 3% warmup -> three flat-top holds with decaying envelope
#           1.55*D -> 1.05*D, each interrupted by one narrow Gaussian dip to
#           0.60*D (deeper than the parent's 0.65*D trough, for a stronger
#           repair contrast) -> proven linear tail from the final hold to
#           exactly gamma_min at the last step.
#   alpha — 0.4*alpha0 exploration floor during the holds (AEP moves freely);
#           Gaussian restoration bursts locked to the lr dips, strengthened
#           4 -> 10 alpha0 across cycles because each burst is briefer than
#           the parent's cosine burst; then the proven bounded endgame
#           (logistic ramp -> 6*alpha0 plateau -> cubic geometric climb to
#           5*alpha0*D/gamma_min).
#   betas — proven transitions; beta1 dips to 0.03 inside each (stronger)
#           burst so momentum cannot carry turbines back over the boundary
#           mid-repair.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three hold/dip cycles inside the exploration phase
_HI0 = 1.55           # first hold level, in units of D — a sustained peak
_HI1 = 1.05           # final hold; the linear tail starts from here (proven)
_LO = 0.60            # dip-bottom lr, in units of D (deeper than parent 0.65)
_SIG = 0.09           # Gaussian dip half-width, as a fraction of one cycle
_A_LO = 0.4           # exploration penalty floor during holds, alpha0 units
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
_B1_BURST = 0.03      # momentum inside each restoration dip
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> flat-top holds with narrow Gaussian dips -> linear tail ---
    # fc freezes at 1 past _F_COOL; pos then freezes at 0, where the Gaussian
    # dip is ~0, so the cool-down starts exactly from the final hold _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    phase = _N_CYC * fc
    pos = phase - jnp.floor(phase)                            # sawtooth in [0, 1) per cycle
    dip = jnp.exp(-0.5 * ((pos - 0.5) / _SIG) ** 2)           # 1 at mid-cycle dip, ~0 on holds
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying hold envelope
    lr_wave = (hi - (hi - _LO) * dip) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_wave - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + dip-locked growing bursts -> plateau -> terminal climb ---
    # The burst IS the Gaussian dip, so restoration coincides exactly with the
    # near-zero step size and vanishes in the tail (dip ~ 0 for frac >= _F_COOL).
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * dip
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-dip beta1 reduction ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_BURST) * dip              # drop momentum while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2