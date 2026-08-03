import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the lineage has
# now tried restarts (SGDR), cyclic alpha, and mid-run restoration bursts —
# every exploration phase so far OSCILLATES. This attempt is the opposite,
# untried corner of the menu: a WSD / ONE-CYCLE "hold" (prior-art §2/§6) with
# an ADMM-STYLE CONSTANT MODERATE PENALTY (explicitly listed as untried).
#
#   lr    — warmup -> HOT SUSTAINED HOLD -> single straight tail. 5% linear
#           warmup, then a gently tilted hold 1.50*D -> 1.15*D until 58%.
#           The parent only *touches* 1.65*D at momentary cosine peaks; its
#           time-averaged exploration lr is ~1.0*D because half of every
#           cycle is spent in 0.65*D troughs. Holding ~1.3-1.5*D for the
#           whole exploration phase is the "higher/LONGER lr peak" the
#           guidance asks for, delivered as integrated hot-lr time rather
#           than a taller spike. From 58% the proven straight linear tail
#           lands exactly on gamma_min at the last step (§6: hold near c*D
#           then (near-)linear cool-down beats cosine/product decay).
#   alpha — ADMM-style CONSTANT 1.8*alpha0 through the entire hold: no floor,
#           no bursts, no coupling to lr. 1.8 matches the parent's *realized
#           average* exploration penalty (floor 0.4 + burst mean ~1.7), so
#           violation debt stays bounded without ever yanking the layout
#           mid-basin-hop — the constant multiplier is the ADMM fixed-rho
#           bet from the untried-directions list. The endgame is the proven
#           5/5-feasible machinery, reinterpreted as an epsilon-constrained
#           shrinking tolerance (§7.9): logistic ramp to the bounded
#           6*alpha0 plateau as the tail begins, then the cubic-delayed
#           geometric climb from 78% contracting the enforced violation band
#           until alpha lands on the proven terminal 5*alpha0*D/gamma_min.
#   betas — beta2 keeps the proven 0.2 -> 0.9 logistic, re-centered on the
#           new cool-down start. beta1 is the one-cycle anti-correlation
#           (§2/§4, untried): LOW momentum (0.05) while lr is hot so single
#           steps stay correctable, a Sutskever-style ramp UP to 0.15 as lr
#           decays (momentum as implicit ALM multiplier lets the moderate
#           constant alpha enforce constraints, and speeds polishing along
#           shallow AEP valleys), then the proven gate down to 0.02 during
#           the terminal alpha spike so momentum never carries turbines back
#           across the boundary.
_F_WARM = 0.05        # linear lr warmup (slightly longer: we go straight to a sustained hot hold)
_F_COOL = 0.58        # hold ends here; straight linear decay to gamma_min at 100%
_HI0 = 1.50           # hold entry lr, in units of D — sustained, not a momentary peak
_HI1 = 1.15           # hold exit lr; the linear tail starts from here
_A_CONST = 1.8        # ADMM-style constant exploration penalty, in alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.62      # logistic alpha ramp centered just after the cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven 5/5-feasible scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HOT = 0.05        # low momentum while lr is hot (one-cycle anti-correlation)
_B1_MID = 0.15        # ramped-up momentum during the cool polish (implicit ALM)
_B1_CENTER = 0.62     # beta1 up-ramp center, once the tail is underway
_B1_WIDTH = 0.06
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike (proven)
_B1_GATE = 0.88
_B1_GWIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot hold -> straight linear tail to gamma_min ---
    h = jnp.clip(frac / _F_COOL, 0.0, 1.0)                    # freezes at 1 past the hold
    hi = (_HI0 + (_HI1 - _HI0) * h) * Dj                      # 1.50*D -> 1.15*D across the hold
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (hi - gmin) * (1.0 - p)                   # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: ADMM constant -> bounded plateau -> terminal shrinking-tolerance climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_CONST + (_A_PLAT - _A_CONST) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven beta2 transition; one-cycle beta1 ramp with terminal gate ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    b1_mid = _B1_HOT + (_B1_MID - _B1_HOT) * b1r              # anti-correlated with lr
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _B1_GATE) / _B1_GWIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * gate                 # kill momentum in the spike

    return lr, alpha, beta1, beta2