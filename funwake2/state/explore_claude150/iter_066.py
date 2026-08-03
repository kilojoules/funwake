import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the top UNTRIED
# row of the prior-art menu — WSD / one-cycle lr (§6, §2): "hold near c*D,
# then (near-)linear cool-down beats cosine/product decay". Every strong
# parent so far explores with COSINE CYCLES that repeatedly fall into cold
# troughs where the layout partially re-freezes between hops; here the
# exploration is a single SUSTAINED HOT HOLD, so integrated exploration heat
# is HIGHER than the burst best (mean ~1.15*D vs ~1.0*D over the hot phase)
# without ever exceeding the proven per-step ceiling.
#
#   lr    — trapezoid with a tilted roof: 3% linear warmup to 1.30*D, then a
#           slow linear "annealed hold" down to 1.00*D at 58% (continuous
#           exploration, no troughs, no restarts), then the proven straight
#           linear tail landing exactly on gamma_min at the last step. The
#           whole schedule is piecewise-linear — zero cycles anywhere.
#   alpha — DECOUPLED, ADMM-flavoured: a constant moderate exploration
#           pressure at 0.5*alpha0 through the entire hold (slightly above
#           the burst-best's 0.4 floor, because there are no mid-run repair
#           bursts — debt is kept small continuously instead of repaid in
#           installments), then the proven single-repayment endgame that was
#           5/5-seed feasible before bursts existed: logistic ramp to the
#           bounded 6*alpha0 ALM plateau at cool-down, and the cubic-delayed
#           geometric climb from 78% landing on the terminal feasibility
#           spike 5*alpha0*D/gamma_min.
#   betas — proven beta2 0.2 -> 0.9 at cool-down, PLUS the untried menu bet:
#           momentum ANTI-CORRELATED with lr (one-cycle / Sutskever rising
#           ramp). beta1 stays at native 0.1 while lr is hot, rises to 0.4 as
#           lr falls (momentum coasts turbines into the minima the hold
#           found, acting as an implicit ALM multiplier), then the proven
#           gate down to 0.02 inside the terminal alpha spike so momentum
#           never fights the feasibility restoration.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.58        # hold ends here; linear decay to gamma_min at 100%
_HI_START = 1.30      # roof of the hold at warmup end, in units of D
_HI_END = 1.00        # roof of the hold at cool-down start (tail starts here)
_A_LO = 0.5           # constant exploration penalty, in alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.62      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_LO_EXPLORE = 0.1  # native momentum while the hold is hot
_B1_COAST = 0.4       # raised momentum during the cool-down (one-cycle)
_B1_RISE_CENTER = 0.70
_B1_RISE_WIDTH = 0.05
_B1_TERM = 0.02       # near-zero momentum during the terminal alpha spike
_B1_GATE_CENTER = 0.88
_B1_GATE_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot hold (1.30*D -> 1.00*D) -> linear tail ---
    # hold_frac freezes at 1 past _F_COOL, so the tail starts exactly from
    # the roof's cool end, _HI_END * D.
    hold_frac = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    lr_hold = (_HI_START + (_HI_END - _HI_START) * hold_frac) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: constant moderate pressure -> plateau -> terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven beta2 transition + rising-then-gated beta1 ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    rise = 1.0 / (1.0 + jnp.exp(-(frac - _B1_RISE_CENTER) / _B1_RISE_WIDTH))
    b1_mid = _B1_LO_EXPLORE + (_B1_COAST - _B1_LO_EXPLORE) * rise
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _B1_GATE_CENTER) / _B1_GATE_WIDTH))
    beta1 = b1_mid + (_B1_TERM - b1_mid) * gate

    return lr, alpha, beta1, beta2