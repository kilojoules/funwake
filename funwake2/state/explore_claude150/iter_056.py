import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the two menu rows
# still untouched anywhere in the lineage, fused into one coherent design —
# a WSD/one-cycle lr (prior-art §2/§6: hold near c*D, then a straight linear
# cool-down — no cosine, no restarts) driven by an EPSILON-CONSTRAINED
# CONTRACTING VIOLATION BAND for alpha (§7.9: schedule the *enforced tolerance*
# gamma(t), not alpha itself, and let it shrink to gamma_min only at the end).
#
#   lr    — trapezoid, not cycles: 4% linear warmup -> a SUSTAINED hot hold at
#           1.35*D until 50% (far more hot-time than any cosine peak in the
#           lineage — the "higher/longer early peak" the parent guidance asks
#           for, delivered as area under the curve rather than a taller spike)
#           -> the proven straight linear tail landing exactly on gamma_min at
#           the last step. Cosine restarts spend most of each cycle below
#           peak; the hold does not.
#   alpha — ONE continuous law replaces floor+bursts+ramp+plateau+spike:
#           alpha = 5*alpha0*D / gamma(t), where the enforced band gamma(t)
#           contracts geometrically from 12.5*D (i.e. unconstrained: alpha
#           starts at the proven 0.4*alpha0 exploration floor) down to exactly
#           gamma_min at the last step. Contraction starts at 25% and is
#           quartically back-loaded, so restoring pressure grows smoothly all
#           through the hot hold (replacing the parent's discrete bursts with
#           continuous repayment), passes the proven few-alpha0 ALM range near
#           80%, and sweeps up to the 5/5-seed-feasible terminal
#           5*alpha0*D/gamma_min — the strong terminal restoration is the
#           band closing, not a bolted-on spike.
#   betas — beta2 keeps the proven 0.2 -> 0.9 transition, re-centered at the
#           new cool-down start (50%). beta1 takes the last untried menu bet
#           (§2/§4, momentum-as-implicit-ALM): anti-correlated with lr — 0.1
#           while hot, RISING to 0.35 mid-decay so momentum integrates the
#           constraint gradient like an ALM multiplier while alpha is still
#           moderate, then the proven terminal gate slams it to 0.02 before
#           the band snaps shut, so momentum never fights the restoration.
_F_WARM = 0.04        # linear lr warmup fraction
_F_HOLD = 0.50        # hot hold ends here; linear decay to gamma_min at 100%
_HI = 1.35            # sustained hold lr, in units of D
_GAIN = 5.0           # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_BAND0 = 12.5         # initial enforced band = 12.5*D -> alpha floor 0.4*alpha0
_F_EPS = 0.25         # band contraction starts here
_P_EPS = 4.0          # quartic back-loading: band stays wide until late
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.50     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_EXPLORE = 0.1     # native low momentum while lr is hot
_B1_PEAK = 0.35       # ALM-style momentum peak mid-decay (anti-phased with lr)
_B1_TERM = 0.02       # proven near-zero momentum during terminal restoration
_B1_RISE_C = 0.68     # momentum ramp center (mid-decay)
_B1_RISE_W = 0.06
_B1_GATE_C = 0.88     # terminal gate center (proven)
_B1_GATE_W = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> sustained hot hold -> straight linear tail (WSD) ---
    r = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (_HI * Dj - gmin) * (1.0 - r)   # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)         # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: epsilon-constrained contracting band ---
    # gamma(t) shrinks geometrically 12.5*D -> gamma_min; alpha is the penalty
    # that enforces the current band: alpha = GAIN*alpha0*D/gamma(t). One law
    # spans exploration floor, mid-run restoration, ALM range, terminal spike.
    u = jnp.clip((frac - _F_EPS) / (1.0 - _F_EPS), 0.0, 1.0) ** _P_EPS
    log_span = jnp.log(jnp.maximum(_BAND0 * Dj / gmin, 1.0))
    gamma_t = _BAND0 * Dj * jnp.exp(-u * log_span)  # ends exactly at gamma_min
    alpha = _GAIN * alpha0 * Dj / gamma_t           # ends at 5*alpha0*D/gmin

    # --- betas: proven beta2 transition; one-cycle beta1 rise + terminal gate ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    rise = 1.0 / (1.0 + jnp.exp(-(frac - _B1_RISE_C) / _B1_RISE_W))
    b1_mid = _B1_EXPLORE + (_B1_PEAK - _B1_EXPLORE) * rise
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _B1_GATE_C) / _B1_GATE_W))
    beta1 = b1_mid + (_B1_TERM - b1_mid) * gate

    return lr, alpha, beta1, beta2