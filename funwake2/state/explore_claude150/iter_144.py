import jax.numpy as jnp

# STRUCTURAL CHANGE vs the WSD flat-top best (+0.0577%): the EXPLORATION
# MOMENTUM SYSTEM is inverted, not the lr waveform.  Every schedule in this
# lineage explored with native low momentum (beta1 = 0.1) and tried to buy
# AEP by reshaping the lr envelope — and the last eight envelope/constant
# tweaks all landed at or below the best.  This candidate keeps the proven
# lr waveform and the 5/5-seed-feasible alpha/beta2 machinery BIT-IDENTICAL
# in the feasibility phase and instead attacks the one untried prior-art
# axis (§4 momentum ramps; synthesis bet #4 "phase-transition the Adam
# moments with the alpha phase"): a DEMON-STYLE HIGH -> NATIVE MOMENTUM ARC.
#
# Mechanism: the skeleton's gradients are stochastic (wind conditions are
# resampled every iteration), so with beta1 = 0.1 each hot step follows an
# almost-raw single-sample gradient and turbines dither.  Holding
# beta1 = 0.85 during the hot flat-top makes the update direction an EMA
# over ~7 wind samples — a larger effective batch exactly where the
# basin-hopping decisions are made — so turbines travel coherently through
# wake shadows instead of jittering in place.  Momentum is an exploration
# device only: it ramps linearly back to the native 0.1 and is EXACTLY 0.1
# before the cool-down begins, so the entire proven feasibility pipeline
# (cool-down at 62%, bounded ALM plateau, terminal spike) runs unchanged.
#
#   lr    — UNCHANGED from the best: 3% linear warmup -> flat-top hold with
#           envelope 1.5*D -> 1.1*D and three narrow sin^12 repair notches
#           down to 0.35*D -> exploration ends at 62% -> straight linear
#           tail landing exactly on gamma_min at the last step.
#   alpha — UNCHANGED: 0.4*alpha0 exploration floor, growing notch-locked
#           restoration bursts (3 -> 8 alpha0), logistic ramp to the bounded
#           6*alpha0 ALM plateau at 66%, cubic-delayed geometric climb from
#           78% to the terminal 5*alpha0*D/gamma_min feasibility spike.
#   beta1 — NEW: 0.85 heavy-smoothing hold while hot, linear decay to the
#           native 0.1 across 42% -> 58% (complete before cool-down), hard
#           dips to 0.05 inside each repair notch (momentum must never carry
#           turbines back across the boundary mid-repair), and the proven
#           gated drop to 0.02 during the terminal alpha spike.
#   beta2 — UNCHANGED: native 0.2 while exploring, logistic rise to 0.9 at
#           the cool-down for the polish/feasibility phase.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three repair notches inside the exploration phase
_Q = 6.0              # notch = sin^(2Q); ~80% of each cycle at full hold lr
_HI0 = 1.5            # initial hold level, in units of D (proven)
_HI1 = 1.1            # final hold level; the linear tail starts from here
_LO = 0.35            # notch-bottom lr — deep, surgical repair windows
_A_LO = 0.4           # exploration penalty floor, in alpha0 units (proven)
_A_B0 = 3.0           # first restoration burst height, in alpha0 units
_A_B1 = 8.0           # last restoration burst height, in alpha0 units
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
_B1_EXPLORE = 0.85    # NEW: heavy noise-smoothing momentum while hot
_B1_NATIVE = 0.1      # native momentum for the whole feasibility phase
_B1_DECAY0 = 0.42     # momentum arc: linear 0.85 -> 0.1 over this window...
_B1_DECAY1 = 0.58     # ...finishing strictly before the 62% cool-down
_B1_NOTCH = 0.05      # reduced momentum inside each repair notch (proven)
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> flat-top hold with three deep narrow notches -> tail ---
    # notch = sin(pi*N*fc)^(2Q): 0 at fc=0 and fc=1 (so the tail launches from
    # the clean hold level _HI1*D), ~0 for ~80% of each cycle, 1 briefly at
    # each cycle midpoint. fc freezes at 1 past _F_COOL.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    notch = jnp.sin(jnp.pi * _N_CYC * fc) ** (2.0 * _Q)
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying hold envelope
    lr_hold = (_LO + (hi - _LO) * (1.0 - notch)) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + notch-synchronized growing bursts -> plateau -> climb ---
    # Bursts fire exactly inside the lr notches (repair when steps are small)
    # and vanish for frac >= _F_COOL; the proven bounded endgame then takes over.
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * notch
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- beta2: proven native -> 0.9 transition at the cool-down ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    # --- beta1: NEW momentum arc + proven notch dips and terminal drop ---
    # Linear (clip) arc hits _B1_NATIVE exactly at _B1_DECAY1 < _F_COOL, so
    # every step of the proven feasibility phase sees the identical beta1.
    arc = jnp.clip((frac - _B1_DECAY0) / (_B1_DECAY1 - _B1_DECAY0), 0.0, 1.0)
    b1_base = _B1_EXPLORE + (_B1_NATIVE - _B1_EXPLORE) * arc  # 0.85 -> 0.1
    b1_exp = b1_base - (b1_base - _B1_NOTCH) * notch          # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2