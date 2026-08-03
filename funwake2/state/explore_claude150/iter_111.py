import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-hold best (+0.0577%): the exploration phase
# is rebuilt as a GEOMETRIC-COOLING "ANNEAL" with CHIRPED repair notches, on
# the bet that the clearest remaining AEP lever is CUMULATIVE HEAT — a hotter
# start, an exponential (not linear) hold envelope, and a longer exploration
# phase — while the 5/5-seed-feasible endgame is preserved untouched.
#
# Three structural changes to the exploration block:
#   1. GEOMETRIC ENVELOPE, HOTTER + LONGER: the hold decays exponentially
#      1.85*D -> 1.05*D (simulated-annealing-style cooling: proportionally
#      more budget at the hottest temperatures than the parent's linear
#      1.5 -> 1.1 ramp) and exploration is extended from 62% to 66%.
#   2. CHIRPED NOTCH CADENCE: four repair notches placed on an accelerating
#      clock g = fc^1.6 instead of the parent's uniform three. The first hot
#      stretch is long (~18% of the run uninterrupted at ~1.85*D — the deep
#      basin hop happens while hottest), then repairs arrive faster and
#      faster as the layout settles and violation debt accrues quicker.
#   3. DEEPENING NOTCHES: the notch floor sinks 0.45*D -> 0.28*D across the
#      phase — early repairs are light touch-ups, late repairs are cold and
#      surgical, matching the growing alpha bursts (3 -> 9 alpha0).
#
# Everything feasibility-critical is the proven machinery, re-anchored to the
# new phase boundary: notch-synchronized growing alpha bursts, logistic ramp
# to the bounded 6*alpha0 ALM plateau just after cool-down (0.70), linear lr
# tail landing exactly on gamma_min, cubic-delayed geometric alpha climb from
# 79% to the terminal 5*alpha0*D/gamma_min spike, beta2 0.2 -> 0.9 at
# cool-down, beta1 0.1 with per-notch dips to 0.05 and the gated terminal
# drop to 0.02.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.66        # exploration extended; linear decay to gamma_min at 100%
_N_CYC = 4.0          # four repair notches on the chirped clock
_CHIRP = 1.6          # notch clock g = fc^1.6: sparse early, dense late
_Q = 6.0              # notch = sin^(2Q); ~80% of each cycle at full hold lr
_HI0 = 1.85           # initial hold level, in units of D — hotter than any tried
_HI1 = 1.05           # final hold level; the linear tail launches from here
_LO0 = 0.45           # first notch-bottom lr, in units of D (light repair)
_LO1 = 0.28           # last notch-bottom lr (deep, surgical repair)
_A_LO = 0.35          # exploration penalty floor at full heat, in alpha0 units
_A_B0 = 3.0           # first restoration burst height, in alpha0 units (proven)
_A_B1 = 9.0           # last restoration burst height — stronger to match heat
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.70      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.035
_F_TERM = 0.79        # terminal geometric alpha climb starts here
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.66     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_NOTCH = 0.05      # reduced momentum inside each repair notch
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.89
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> geometric-cooling hold with chirped deepening notches
    #         -> linear tail landing exactly on gamma_min ---
    # notch = sin(pi*N*g)^(2Q) with g = fc^CHIRP: 0 at fc=0 and fc=1 (so the
    # tail launches from the clean hold level _HI1*D), ~0 for ~80% of each
    # cycle, 1 briefly at each notch center; centers accelerate as fc -> 1.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    g = fc ** _CHIRP
    notch = jnp.sin(jnp.pi * _N_CYC * g) ** (2.0 * _Q)
    hi = _HI0 * (_HI1 / _HI0) ** fc                           # exponential hold envelope
    lo = _LO0 + (_LO1 - _LO0) * fc                            # notches deepen over time
    lr_hold = (lo + (hi - lo) * (1.0 - notch)) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + chirp-synchronized growing bursts -> plateau -> climb ---
    # Bursts fire exactly inside the lr notches (repair when steps are small)
    # and vanish for frac >= _F_COOL; the proven bounded endgame then takes over.
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * notch
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-notch beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_NOTCH) * notch            # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2