import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-hold best (+0.0577%): the repair-notch /
# restoration-burst cadence is REMOVED entirely and replaced by a SUMT-STYLE
# CONTINUATION LADDER — the classical sequential-penalty structure (SUMT /
# ALM outer iterations, prior-art §7.9 "dynamic penalty" + §6 step decay)
# that no attempt has tried: a staircase of joint (lr, alpha) stages where
# each stage is a fixed subproblem — lr held CONSTANT per rung, alpha held
# CONSTANT per rung — and every stage transition simultaneously steps lr DOWN
# geometrically and steps alpha UP geometrically. No mid-run alpha bursts, no
# lr dips: constraint pressure is a monotone ratchet (ADMM-style moderate
# constant penalty per stage), so feasibility debt is never re-incurred after
# a repair, and the lr never wastes budget below its current hold level.
#
# The ladder is also HOTTER cumulatively than the winning hold: rungs at
# 1.70, 1.45, 1.23, 1.04 (*D) average ~1.36*D across exploration vs ~1.1*D
# effective for the notched hold — pushing further along the direction that
# produced the last two improvements (more sustained heat), while the
# near-doubled exploration alpha floor (0.7*alpha0 vs 0.4) plus the monotone
# ratchet compensates for losing the burst-repair windows.
#
#   lr    — 3% linear warmup (proven) -> 4-rung geometric staircase of flat
#           holds 1.7*D -> 1.04*D (smooth logistic stage transitions) ->
#           exploration ends at the proven 62% -> proven straight linear tail
#           landing exactly on gamma_min at the last step.
#   alpha — geometric ratchet 0.7 -> 2.8 alpha0 locked to the SAME stage
#           transitions (each colder subproblem is a stricter subproblem),
#           then the proven logistic lift to the bounded 6*alpha0 ALM plateau
#           at 66%, and the proven 5/5-seed-feasible cubic-delayed geometric
#           climb from 78% to the terminal 5*alpha0*D/gamma_min spike.
#   betas — beta2 0.2 -> 0.9 at cool-down (proven). beta1 anti-correlated
#           with the lr ladder (one-cycle, §2): 0.06 on the hottest rung
#           rising to 0.12 on the coldest, then the proven gated drop to
#           0.02 during the terminal alpha spike.
_K = 4.0              # number of ladder stages inside the exploration phase
_W = 0.06             # logistic width of each stage transition (in rung units)
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_LR_HI = 1.7          # first-rung hold level, in units of D — hottest stage
_LR_R = 0.85          # geometric lr ratio per rung: 1.70 -> 1.45 -> 1.23 -> 1.04
_A_LO = 0.7           # first-rung penalty, in alpha0 units — ADMM-moderate, no dips
_A_R = 4.0 ** (1.0 / 3.0)   # geometric alpha ratio per rung: 0.7 -> 2.8 over 3 steps
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha lift centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HOT = 0.06        # low momentum on the hottest rung (one-cycle coupling)
_B1_COLD = 0.12       # higher momentum on the coldest rung
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- smooth stage counter: 0 -> K-1 via logistic steps at x = 1, 2, 3 ---
    # rung is ~integer inside each stage and transitions smoothly at the
    # thresholds; it freezes at K-1 past _F_COOL, so the linear tail launches
    # from the clean coldest hold level 1.04*D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    x = _K * fc
    rung = (1.0 / (1.0 + jnp.exp(-(x - 1.0) / _W))
            + 1.0 / (1.0 + jnp.exp(-(x - 2.0) / _W))
            + 1.0 / (1.0 + jnp.exp(-(x - 3.0) / _W)))
    rungf = rung / (_K - 1.0)                # 0 (hottest stage) -> 1 (coldest)

    # --- lr: warmup -> geometric staircase of flat holds -> linear tail ---
    lr_ladder = _LR_HI * (_LR_R ** rung) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_ladder - gmin) * (1.0 - p)            # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: monotone per-stage ratchet -> plateau lift -> terminal climb ---
    # Each lr rung-down fires the matching alpha rung-up: colder, stricter
    # subproblems in lockstep. Past cool-down the ratchet sits at a_top and
    # the proven bounded endgame takes over.
    a_top = _A_LO * (_A_R ** (_K - 1.0))
    base = _A_LO * (_A_R ** rung)
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = base + (_A_PLAT - a_top) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven beta2 transition + ladder-coupled beta1 ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HOT + (_B1_COLD - _B1_HOT) * rungf           # momentum rises as lr cools
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2