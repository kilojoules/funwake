import jax.numpy as jnp

# STRUCTURALLY NEW vs both the flat-top-notch best (+0.0577%) and the
# one-cycle parent: this is classical SUMT — the Fiacco–McCormick
# SEQUENTIAL PENALTY METHOD — mapped open-loop onto the step budget.
# The run becomes K=4 discrete SUBPROBLEMS. Within each subproblem BOTH
# lr and alpha are HELD CONSTANT (a true WSD-style flat hold, §6 — but
# repeated as a ladder, a waveform never tried: not cosine, not one-cycle,
# not notched, not linear). Between subproblems lr drops by a fixed
# geometric factor and alpha RISES by a fixed geometric factor — the
# textbook penalty-continuation loop (§7.9 as discrete continuation
# rather than a smooth power law). Each rung gives the layout time to
# EQUILIBRATE at one (temperature, stiffness) pair before the next
# tightening — the annealing-ladder property no smooth schedule has.
#
#   lr    — 3% linear warmup (proven), then four flat rungs at
#           1.8 / 1.30 / 0.94 / 0.68 * D spanning the proven exploration
#           window [0, 62%] (front-loaded: hotter than the best's hold
#           early, cumulative heat slightly above it), then the PROVEN
#           straight linear tail landing exactly on gamma_min at the
#           last step.
#   alpha — per-rung constants 0.35 / 0.90 / 2.32 / 6.0 * alpha0: the
#           freest start yet during the hottest rung, geometric
#           tightening at every rung switch, arriving at the proven
#           bounded 6*alpha0 ALM plateau for the whole cool-down, then
#           the 5/5-seed-feasible cubic-delayed geometric climb from 78%
#           to the terminal 5*alpha0*D/gamma_min spike — feasibility
#           endgame unchanged.
#   beta1 — 0.1 (native, proven) with a brief Gaussian dip to 0.04 just
#           AFTER each rung switch: momentum built from 1.8*D-scale steps
#           must not be carried across a simultaneous lr-drop/alpha-jump
#           (stale-direction flush, the discrete analogue of the best's
#           in-notch dip); gated to the proven 0.02 during the terminal
#           alpha spike.
#   beta2 — the proven 0.2 -> 0.9 logistic transition at cool-down start.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # ladder ends here; proven linear tail to gamma_min
_K = 4.0              # number of SUMT subproblems (rungs), equal length
_LR_HI = 1.8          # top-rung lr in units of D — hottest sustained hold yet
_LR_LO = 0.68         # bottom-rung lr in units of D (tail launches from here)
_R = (_LR_LO / _LR_HI) ** (1.0 / (_K - 1.0))   # per-rung lr contraction
_A_LO = 0.35          # alpha on the hot rung, in alpha0 units (freest start)
_A_PLAT = 6.0         # alpha on the last rung = proven bounded ALM plateau
_S = (_A_PLAT / _A_LO) ** (1.0 / (_K - 1.0))   # per-rung penalty growth
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum inside each subproblem (proven)
_B1_DIP = 0.04        # momentum right after a rung switch (stale-flush)
_U_W = 0.10           # dip width, in within-rung progress units
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- SUMT clock: rung index k (frozen at K-1 past cool-down) and
    #     within-rung progress u in [0, 1] ---
    fc = jnp.clip(frac / _F_COOL, 0.0, 1.0)
    k = jnp.minimum(jnp.floor(fc * _K), _K - 1.0)
    u = jnp.clip(fc * _K - k, 0.0, 1.0)      # sticks at 1 once the ladder ends

    # --- lr: geometric ladder of flat holds -> proven linear tail ---
    lr_rung = _LR_HI * (_R ** k) * Dj                       # constant per rung
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_rung - gmin) * (1.0 - p)            # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                 # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: per-rung constant penalty continuation -> plateau -> spike ---
    alpha_units = _A_LO * (_S ** k)                         # 0.35 -> 6 alpha0, stepwise
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)    # ends at 5*alpha0*D/gmin

    # --- beta1: post-switch stale-momentum flush + terminal gate ---
    dip = jnp.exp(-((u / _U_W) ** 2))                       # ~1 just after a switch
    b1_exp = _B1_HI - (_B1_HI - _B1_DIP) * dip
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    # --- beta2: proven low -> high logistic transition at cool-down ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    return lr, alpha, beta1, beta2