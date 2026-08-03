import jax.numpy as jnp

# STRUCTURALLY NEW vs the notched flat-top best (+0.0577%): the first
# schedule with NO oscillation anywhere — the duty-cycle trend taken to its
# limit. The lineage cosine restarts (+0.0428) -> anti-phased bursts
# (+0.0533) -> 80%-duty flat-top with repair notches (+0.0577) monotonically
# rewarded MORE sustained heat; here duty = 100%: a pure WSD hold (menu §6:
# "hold near c*D, then (near-)linear cool-down") with mid-run repair windows
# DELETED and their job handed to two explicitly-untried mechanisms:
#
#   alpha — ADMM-STYLE CONSTANT MODERATE PENALTY (untried menu direction):
#           a flat 1.0*alpha0 through the whole exploration phase — no
#           floor-plus-bursts, no coupling to lr. Violation debt is bounded
#           continuously instead of repaid inside notch windows, so not one
#           exploration step is spent at crippled repair-lr. The proven
#           endgame is kept unchanged: logistic ramp to the bounded 6*alpha0
#           ALM plateau at 66%, then the cubic-delayed geometric climb from
#           78% to the 5/5-seed-feasible terminal 5*alpha0*D/gamma_min.
#   beta1 — MOMENTUM AS IMPLICIT ALM MULTIPLIER (§4, untried): a smooth
#           Sutskever-style rise 0.1 -> 0.3 across mid-exploration. Momentum
#           integrates the constant penalty gradient across steps like a
#           running multiplier estimate, letting the MODERATE alpha enforce
#           constraints the old schedules needed 8*alpha0 bursts for — while
#           also carrying coherent long-range AEP drift at full heat. It
#           hands back the native 0.1 before the polish and gates to the
#           proven 0.02 during the terminal spike, so the diverging alpha
#           never rides momentum.
#   lr    — proven pieces only: 3% linear warmup, uninterrupted hold with
#           envelope decaying 1.5*D -> 1.15*D (cumulatively hotter than any
#           waveform tried — every exploration step sits at the full
#           envelope), cool-down at the proven 62%, straight linear tail
#           landing exactly on gamma_min at the last step.
#   beta2 — proven 0.2 -> 0.9 logistic transition at the cool-down start.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_HI0 = 1.5            # initial hold level, in units of D
_HI1 = 1.15           # final hold level; the linear tail starts from here
_A_EXP = 1.0          # ADMM-style constant exploration penalty, in alpha0 units
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
_B1_BASE = 0.1        # native momentum at the hot start and through the polish
_B1_PEAK = 0.3        # implicit-ALM momentum plateau across mid-exploration
_B1_RISE_C = 0.18     # momentum ramps up once the layout is moving
_B1_RISE_W = 0.05
_B1_FALL_C = 0.55     # and hands back native momentum before the cool-down
_B1_FALL_W = 0.04
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> uninterrupted decaying hold -> linear tail ---
    # fc freezes at 1 past _F_COOL, so the tail launches from the clean
    # final hold level _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    lr_hold = (_HI0 + (_HI1 - _HI0) * fc) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: ADMM constant -> logistic ramp -> bounded plateau -> climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_EXP + (_A_PLAT - _A_EXP) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- beta1: mid-exploration momentum plateau -> native polish -> gate ---
    rise = 1.0 / (1.0 + jnp.exp(-(frac - _B1_RISE_C) / _B1_RISE_W))
    fall = 1.0 / (1.0 + jnp.exp(-(frac - _B1_FALL_C) / _B1_FALL_W))
    b1_exp = _B1_BASE + (_B1_PEAK - _B1_BASE) * rise * (1.0 - fall)
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * gate

    # --- beta2: proven transition ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    return lr, alpha, beta1, beta2