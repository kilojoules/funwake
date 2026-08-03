import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the exploration
# phase is rebuilt around the two menu directions the lineage has never run —
# a WSD SUSTAINED-HOT PLATEAU lr (menu §6/§2: "hold near c*D, then near-linear
# cool-down") and an ADMM-STYLE CONSTANT MODERATE PENALTY (search-state
# direction explicitly listed as untried). The proven endgame — linear tail to
# gamma_min, logistic alpha ramp to the bounded 6*alpha0 plateau, cubic-delayed
# geometric climb to 5*alpha0*D/gamma_min, beta2 0.2->0.9 and beta1 ->0.02 —
# is preserved verbatim, since it restored 5/5-seed feasibility both with and
# without mid-run repair in prior generations.
#
#   lr    — the parent spends much of exploration at/near 0.65*D troughs where
#           basin hops stall; here there are NO troughs. After the proven 3%
#           warmup, lr holds a hot plateau of mean 1.25*D for the whole
#           exploration phase (time-average ~25% hotter than the restart
#           schedule) with a gentle +/-0.15*D cosine dither (3.5 waves) so the
#           dynamics never lock into a limit cycle. The half-integer dither
#           count makes the plateau END at its coolest point, 1.10*D, so the
#           proven straight linear tail starts near the proven 1.05*D and
#           lands exactly on gamma_min at the last step.
#   alpha — ADMM-flavored: a CONSTANT moderate 1.5*alpha0 through the entire
#           hot phase. No floor/burst modulation: the penalty is high enough
#           to keep violation debt bounded while lr stays hot for 62% of the
#           run, low enough that basin hops still trade violation for AEP.
#           The single debt repayment is the proven endgame: logistic ramp
#           (center 0.66) to the bounded 6*alpha0 ALM plateau, then the
#           cubic-delayed geometric climb from 78% onto the 5/5-seed-feasible
#           terminal 5*alpha0*D/gamma_min.
#   beta1 — the one axis no parent has touched: a Sutskever INCREASING
#           momentum ramp 0.1 -> 0.3 across exploration (momentum as implicit
#           ALM multiplier, menu bet; amplification ~1.4x keeps the effective
#           late-exploration drift near the tried 1.65*D peak without raising
#           lr itself). At cool-down momentum returns to the proven 0.1 for
#           polishing, then gates down to 0.02 inside the terminal spike so it
#           never fights the feasibility restoration.
#   beta2 — proven transition: 0.2 (fast, sign-like) while exploring,
#           logistic to 0.9 at the cool-down boundary.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_PLAT = 1.25          # sustained plateau mean, in units of D
_DITH = 0.15          # dither amplitude around the plateau (lr in [1.10, 1.40]*D)
_N_DITH = 3.5         # half-integer -> plateau starts at 1.40*D, ends at 1.10*D
_A_EXP = 1.5          # ADMM-style constant exploration penalty, in alpha0 units
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
_B1_EXP0 = 0.1        # momentum at the start of exploration (native)
_B1_EXP1 = 0.3        # Sutskever ramp target at the end of exploration
_B1_POLISH = 0.1      # proven polish momentum
_B1_COOL_CENTER = 0.62
_B1_COOL_WIDTH = 0.05
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> sustained hot dithered plateau -> proven linear tail ---
    # fc freezes at 1 past _F_COOL; cos(2*pi*3.5) = -1 pins the cool-down
    # start at the plateau's coolest point, (_PLAT - _DITH) * D = 1.10 * D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    lr_plat = (_PLAT + _DITH * jnp.cos(2.0 * jnp.pi * _N_DITH * fc)) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_plat - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: ADMM constant -> logistic bounded plateau -> terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_EXP + (_A_PLAT - _A_EXP) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: Sutskever exploration ramp + proven cool-down/terminal gates ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_ramp = _B1_EXP0 + (_B1_EXP1 - _B1_EXP0) * fc           # rising momentum while hot
    b1c = 1.0 / (1.0 + jnp.exp(-(frac - _B1_COOL_CENTER) / _B1_COOL_WIDTH))
    b1_mid = b1_ramp + (_B1_POLISH - b1_ramp) * b1c           # back to 0.1 for polishing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * b1r                  # 0.02 in the terminal spike

    return lr, alpha, beta1, beta2