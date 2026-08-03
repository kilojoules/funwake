import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased SGDR best (+0.0533%): the lr backbone is
# no longer cyclic at all. This is the untried WSD / one-cycle bet from the
# prior-art menu (§6, §2): warmup -> a LONG HOT HOLD near c*D -> near-linear
# cool-down to gamma_min. Cosine restarts spend most of the exploration phase
# far below their peaks (parent's time-average lr ~1.0*D despite a 1.65*D
# peak); a tilted flat hold at 1.50*D -> 1.10*D keeps the time-average near
# ~1.3*D — a ~30% hotter exploration *budget* than any lineage attempt, while
# the instantaneous peak (1.50*D) stays inside the proven-safe range. More
# time at temperature, not a taller spike: the direction the guidance asks
# for, delivered by a different mechanism than another peak tweak.
#
#   lr    — 3% warmup; hold tilting 1.50*D -> 1.10*D until 62%; then the
#           proven straight linear tail landing exactly on gamma_min at the
#           last step. Punched into the hold are two NARROW Gaussian repair
#           NOTCHES (~4% of the run each) where lr briefly drops to ~25% of
#           the hold — the parent's proven "repair only at low lr" mechanism,
#           but as brief punctuation instead of broad cosine troughs, so
#           almost no hot-phase step budget is sacrificed.
#   alpha — floor at 0.35*alpha0 through the hold (basin hops trade violation
#           for AEP freely), with GROWING repair bursts (4 -> 8 alpha0 units)
#           co-located with the lr notches — slightly wider than the notches
#           so the penalty brackets each dip. Debt is repaid mid-run exactly
#           when the step size cannot destroy AEP structure. The entire
#           5/5-seed-feasible endgame is preserved verbatim: logistic ramp to
#           the bounded 6*alpha0 ALM plateau after 62%, then the cubic-
#           delayed geometric climb from 78% landing on the terminal
#           5*alpha0*D/gamma_min feasibility spike.
#   betas — proven transitions kept: beta2 0.2 -> 0.9 at cool-down start,
#           beta1 gated 0.1 -> 0.02 in the terminal spike, and beta1 dipping
#           to 0.05 inside each repair notch so momentum never carries
#           turbines back across the boundary while the burst pulls them in.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # hold ends here; linear decay to gamma_min at 100% (proven)
_HI_START = 1.50      # hold entry lr, in units of D — hot but below tried 1.65 peak
_HI_END = 1.10        # hold exit lr; tail starts here (matches proven ~1.05*D)
_NOTCH_C0 = 0.25      # first repair notch center (fraction of run)
_NOTCH_C1 = 0.47      # second repair notch center
_NOTCH_W = 0.018      # lr notch width — narrow, ~4% of the run at 2 sigma
_NOTCH_DEPTH = 0.75   # lr drops to 25% of the hold at each notch center
_A_LO = 0.35          # exploration penalty floor, in alpha0 units
_A_AMP0 = 4.0         # first repair burst height, in alpha0 units
_A_AMP1 = 8.0         # second repair burst height — repairs strengthen
_A_W = 0.022          # burst width — slightly wider than the lr notch
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
_B1_NOTCH = 0.05      # reduced momentum inside each repair notch
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot hold with narrow repair notches -> linear tail ---
    fh = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    hold = (_HI_START + (_HI_END - _HI_START) * fh) * Dj      # 1.50*D -> 1.10*D

    n1 = jnp.exp(-(((frac - _NOTCH_C0) / _NOTCH_W) ** 2))
    n2 = jnp.exp(-(((frac - _NOTCH_C1) / _NOTCH_W) ** 2))
    notch = jnp.clip(n1 + n2, 0.0, 1.0)                       # 1 at notch centers
    lr_hold = hold * (1.0 - _NOTCH_DEPTH * notch)

    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + growing repair bursts -> plateau -> terminal climb ---
    # bursts are co-located with the lr notches (slightly wider), so restoration
    # always coincides with a near-zero step size; their Gaussian tails are
    # negligible by _F_COOL, where the proven logistic ramp takes over.
    r1 = jnp.exp(-(((frac - _NOTCH_C0) / _A_W) ** 2))
    r2 = jnp.exp(-(((frac - _NOTCH_C1) / _A_W) ** 2))
    repair = _A_AMP0 * r1 + _A_AMP1 * r2
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + repair
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-notch beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    rep_shape = jnp.clip(r1 + r2, 0.0, 1.0)
    b1_exp = _B1_HI - (_B1_HI - _B1_NOTCH) * rep_shape        # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2