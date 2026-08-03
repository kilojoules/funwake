import jax.numpy as jnp

# STRUCTURALLY NEW vs both the trapezoid parent (+0.0397%) and the SGDR-burst
# best (+0.0533%). Neither the lr shape nor the alpha machinery is where this
# design bets — those reuse only PROVEN elements. The bet is the one axis the
# entire lineage has left untouched: the MOMENT SCHEDULE. Every attempt so far
# lives in TopFarm's low-momentum regime (beta1<=0.1, beta2<=0.9); this design
# phase-transitions into FULL STANDARD ADAM (0.9/0.999) for a heavy-ball
# polish phase (menu bet 4 + Sutskever increasing-momentum ramp, §4, untried).
#
#   lr    — proven trapezoid, rebalanced to feed the new polish engine:
#           3% warmup -> hot hold at 1.25*D (hotter than the parent's 1.12*D)
#           but cut EARLY at 52%, then the proven straight LINEAR decay
#           landing exactly on gamma_min. The cool-down — 48% of the budget,
#           the longest polish in the lineage — is where heavy-ball momentum
#           converts the hotter hold's disorder into AEP.
#   alpha — proven decoupled pipeline (floor -> logistic ALM plateau ->
#           terminal climb), plus ONE mid-hold restoration notch (§7.5): a
#           narrow Gaussian lr dip to 0.5*D at 30% anti-phased with a
#           5*alpha0 repair burst. A single mid-run debt repayment (between
#           the parent's zero and the best's three) keeps the layout near-
#           feasible entering the polish without wasting exploration budget.
#           Terminal: cubic-delayed geometric climb from 80% landing on the
#           5/5-seed-feasible 5*alpha0*D/gamma_min at the final step.
#   betas — the novelty. Exploration keeps native 0.1/0.2 (proven for basin
#           hopping); at the cool-down start beta1 ramps 0.1 -> 0.9 and beta2
#           0.2 -> 0.999, so the shrinking lr is compensated by ~10x momentum
#           accumulation along consistent wake-gradient valleys — sustained
#           mobility exactly where linear decay beat cosine for that reason.
#           At 88% BOTH unwind (beta1 -> 0.02, beta2 -> 0.9, the proven
#           terminal pair) so no stored momentum can carry turbines across
#           the boundary while the terminal alpha spike collects feasibility.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_C_HOLD = 1.25        # hot hold at 1.25*D — hotter than the parent trapezoid
_F_COOL = 0.52        # early cool-down start: 48% of budget for the polish
_F_NOTCH = 0.30       # single mid-hold restoration event, centered at 30%
_W_NOTCH = 0.03       # narrow Gaussian width (a few % of the run)
_NOTCH_DEPTH = 0.6    # lr dips to 0.4 * hold = 0.5*D at the notch center
_A_NOTCH = 5.0        # repair burst height at the notch, in alpha0 units
_A_LO = 0.4           # exploration penalty floor (proven), in alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau (proven), in alpha0 units
_A_CENTER = 0.60      # logistic alpha ramp engages just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.80        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B1_EXPLORE = 0.1     # native momentum while exploring (proven)
_B1_NOTCH = 0.05      # momentum dip inside the repair notch (proven idea)
_B1_POLISH = 0.9      # heavy-ball standard-Adam polish — THE untried regime
_B1_TERM = 0.02       # near-zero momentum during the terminal spike (proven)
_B2_LO = 0.2          # native adaptive decay while exploring (proven)
_B2_POLISH = 0.999    # standard Adam second moment for the polish
_B2_TERM = 0.9        # proven terminal beta2
_B_CENTER = 0.56      # moment phase transition just after the cool-down start
_B_WIDTH = 0.04
_F_UNWIND = 0.88      # both moments unwind here, before alpha diverges hard
_W_UNWIND = 0.025


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # single mid-hold restoration event: ~1 at the notch center, ~0 elsewhere
    g = jnp.exp(-0.5 * ((frac - _F_NOTCH) / _W_NOTCH) ** 2)

    # --- lr: warmup -> hold at 1.25*D (one notch) -> LINEAR decay to gamma_min ---
    lr_hold = _C_HOLD * Dj * (1.0 - _NOTCH_DEPTH * g)
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)
    lr = lr_env * warm

    # --- alpha: floor + one repair burst -> logistic plateau -> terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + _A_NOTCH * g
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: native explore -> standard-Adam polish -> proven terminal pair ---
    rise = 1.0 / (1.0 + jnp.exp(-(frac - _B_CENTER) / _B_WIDTH))
    unwind = 1.0 / (1.0 + jnp.exp(-(frac - _F_UNWIND) / _W_UNWIND))

    b1_explore = _B1_EXPLORE - (_B1_EXPLORE - _B1_NOTCH) * g  # dip while repairing
    beta1 = b1_explore + (_B1_POLISH - b1_explore) * rise
    beta1 = beta1 + (_B1_TERM - beta1) * unwind

    beta2 = _B2_LO + (_B2_POLISH - _B2_LO) * rise
    beta2 = beta2 + (_B2_TERM - beta2) * unwind

    return lr, alpha, beta1, beta2