import jax.numpy as jnp

# STRUCTURALLY NEW vs both the SGDR-restart lineage and the anti-phased-burst
# best: this is the top UNTRIED lr row of the prior-art menu — WSD / ONE-CYCLE
# ("hold near c*D, then near-linear cool-down beats cosine/product decay",
# §6/§2) — plus the untried beta1 row (one-cycle momentum anti-correlated
# with lr, §2/§4). Every lineage member so far spends half its exploration
# budget deep in cosine troughs (0.65*D); here exploration HOLDS HOT the
# whole time. The proven feasibility machinery is preserved unchanged.
#
#   lr    — WSD WITH HIGH-FREQUENCY DITHER: 3% linear warmup (proven), then a
#           FLAT-HOT stable phase at 1.30*D carrying a fast small cosine
#           dither (8.5 mini-cycles, amplitude 0.25*D → lr sweeps 1.05*D to
#           1.55*D). Mean exploration lr is ~1.30*D — far hotter than any
#           restart scheme's ~1.0*D average — while the extremes (1.55*D
#           peak, 1.05*D floor) are each individually proven safe in the
#           lineage. The half-integer cycle count ends the dither exactly at
#           its 1.05*D trough, from which the proven straight linear tail
#           lands lr exactly on gamma_min at the last step. Structurally:
#           sustained-hot basin exploration with rapid shallow perturbation,
#           instead of a few deep restarts.
#   alpha — proven decoupled shape, untouched: 0.45*alpha0 exploration floor
#           (slightly above 0.4 to cover the hotter average lr's extra
#           violation debt), ONE logistic ramp at the cool-down onto the
#           bounded 6*alpha0 ALM plateau (debt repaid continuously while lr
#           is still effective), then the cubic-delayed geometric climb from
#           78% onto the 5/5-seed-feasible terminal 5*alpha0*D/gamma_min.
#   beta1 — ONE-CYCLE ANTI-CORRELATION (untried anywhere in the lineage):
#           low momentum when lr is hot (0.06 at dither peaks, 0.12 at
#           troughs), then RISING linearly with tail progress toward 0.30 as
#           lr shrinks — momentum acting as an implicit ALM multiplier so
#           the moderate plateau alpha enforces constraints (menu bet 4) —
#           and finally the proven logistic gate down to 0.02 during the
#           terminal alpha spike so diverging alpha never rides momentum.
#   beta2 — proven transition only: 0.2 -> 0.9 logistic at the cool-down.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.60        # stable-hot phase ends here; linear decay to gamma_min
_M = 8.5              # dither mini-cycles; half-integer -> ends at the trough
_BASE = 1.30          # flat hot hold, in units of D
_AMP = 0.25           # dither amplitude: lr in [1.05, 1.55]*D, both proven
_A_LO = 0.45          # exploration penalty floor, in alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.61      # logistic alpha ramp centered at the cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_MID = 0.09        # exploration beta1 midpoint (dither swings +/- 0.03)
_B1_SWING = 0.03      # anti-correlated with lr: 0.06 hot, 0.12 cool
_B1_MAX = 0.30        # one-cycle momentum ceiling as lr shrinks in the tail
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.025


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> flat-hot hold with fast shallow dither -> linear tail ---
    # fc freezes at 1 past _F_COOL; cos(2*pi*8.5) = -1 pins the tail start at
    # the dither trough, 1.05*D (the proven tail-start value).
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    cosd = jnp.cos(2.0 * jnp.pi * _M * fc)                    # fast dither, in [-1, 1]
    lr_hot = (_BASE + _AMP * cosd) * Dj                       # sweeps [1.05, 1.55]*D
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hot - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor -> logistic ramp -> bounded plateau -> terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp            # in alpha0 units
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- beta2: proven cool-down transition ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    # --- beta1: one-cycle anti-correlation with lr, then terminal gate ---
    # Exploration: low momentum at hot dither peaks, higher at cool troughs.
    # Tail: cosd is frozen at -1, so the base sits at 0.12 and rises linearly
    # with tail progress toward _B1_MAX as lr shrinks (momentum up, lr down).
    b1_base = _B1_MID - _B1_SWING * cosd
    b1_cycle = b1_base + (_B1_MAX - b1_base) * p
    b1g = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_cycle + (_B1_LO - b1_cycle) * b1g

    return lr, alpha, beta1, beta2