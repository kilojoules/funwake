import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-notch best (+0.0577%): the DUTY-CYCLE
# INVERSION that beat cosine restarts during exploration is now applied to
# the POLISH TAIL as well — the whole schedule becomes a TWO-TIER WSD
# ("hold, drop, hold, land"). The best spends its final 38% sliding linearly
# through every lr scale from 1.1*D down to gamma_min, so no scale gets more
# than a passing visit; here that slide is restructured into a fast
# half-cosine drop to a dedicated FINE-POLISH HOLD at 0.25*D (14% of the
# budget concentrated at the one scale where turbines make their final
# wake-trade micro-moves), followed by a short linear landing that still
# hits gamma_min exactly at the last step. The freed budget also lets the
# proven hot flat-top run slightly hotter and longer (1.55*D start, 65% end).
#
# Second structural change (menu bets §2/§4, untried anywhere in the
# lineage): ONE-CYCLE MOMENTUM ANTI-CORRELATION. Gradients here are
# deterministic, so the fine hold is a heavy-ball opportunity: beta1 ramps
# UP to 0.5 inside the 0.25*D hold (effective step ~2x along consistent
# descent directions — acceleration through the ill-conditioned final basin
# without raising the raw step size), then the proven gate slams it to 0.02
# before the terminal alpha spike so momentum never fights the restoration.
#
#   lr    — 3% linear warmup (proven) -> flat-top hold with envelope decaying
#           1.55*D -> 1.10*D across three narrow sin^12 repair notches down
#           to 0.35*D, exploration ending at 65% -> fast half-cosine drop to
#           0.25*D by 74% -> FINE-POLISH HOLD at 0.25*D until 88% -> linear
#           landing exactly on gamma_min at the last step.
#   alpha — proven feasibility machinery intact: 0.4*alpha0 exploration
#           floor, growing notch-synchronized restoration bursts (3 -> 8
#           alpha0), logistic ramp to the bounded 6*alpha0 ALM plateau as the
#           drop begins, and the 5/5-seed-feasible cubic-delayed geometric
#           climb from 78% to the terminal 5*alpha0*D/gamma_min spike. Note
#           the plateau/spike now act at SMALLER lr than in the best (0.25*D
#           hold vs a ~0.4-1.1*D slide), so restoration disturbs less AEP
#           structure while remaining mobile enough (60 m steps) to repair.
#   betas — beta2 0.2 -> 0.9 aligned with the drop (proven); beta1 0.1 with
#           dips to 0.05 inside each repair notch (proven), the NEW ramp up
#           to 0.5 inside the fine-polish hold, then the proven gated drop
#           to 0.02 for the terminal alpha spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_EXP = 0.65         # hot flat-top exploration ends here (was 0.62)
_F_HOLD = 0.74        # fast drop ends; fine-polish hold begins
_F_LAND = 0.88        # linear landing onto gamma_min begins
_N_CYC = 3.0          # three repair notches inside the exploration phase
_Q = 6.0              # notch = sin^(2Q); ~80% of each cycle at full hold lr
_HI0 = 1.55           # initial hold level, in units of D — sustained heat
_HI1 = 1.10           # final hold level; the drop launches from here
_LO = 0.35            # notch-bottom lr — deep, surgical repair windows
_MID = 0.25           # fine-polish hold level, in units of D
_A_LO = 0.4           # exploration penalty floor at full heat, in alpha0 units
_A_B0 = 3.0           # first restoration burst height, in alpha0 units (proven)
_A_B1 = 8.0           # last restoration burst height, in alpha0 units (proven)
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.69      # logistic alpha ramp centered just after the drop starts
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.65     # beta2 transition aligned with the start of the drop
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring
_B1_NOTCH = 0.05      # reduced momentum inside each repair notch
_B1_POL = 0.5         # heavy-ball momentum inside the fine-polish hold (NEW)
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_UP_CENTER = 0.76  # momentum ramps up as the fine hold begins
_B1_UP_WIDTH = 0.02
_B1_DN_CENTER = 0.89  # momentum gated off before the spike gets strong
_B1_DN_WIDTH = 0.02


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr tier 1: warmup -> flat-top hold with three deep narrow notches ---
    # notch = sin(pi*N*fc)^(2Q): 0 at fc=0 and fc=1 (the drop launches from
    # the clean hold level _HI1*D), ~0 for ~80% of each cycle, 1 briefly at
    # each cycle midpoint. fc freezes at 1 past _F_EXP, killing the notches.
    fc = jnp.clip(frac, 0.0, _F_EXP) / _F_EXP
    notch = jnp.sin(jnp.pi * _N_CYC * fc) ** (2.0 * _Q)
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying hold envelope
    lr_exp = (_LO + (hi - _LO) * (1.0 - notch)) * Dj

    # --- lr tier 2: fast half-cosine drop -> fine-polish hold -> landing ---
    d = jnp.clip((frac - _F_EXP) / (_F_HOLD - _F_EXP), 0.0, 1.0)
    w = 0.5 * (1.0 + jnp.cos(jnp.pi * d))                     # 1 -> 0 across the drop
    lr_mid = _MID * Dj + (lr_exp - _MID * Dj) * w             # holds at 0.25*D after 74%
    p = jnp.clip((frac - _F_LAND) / (1.0 - _F_LAND), 0.0, 1.0)
    lr_env = gmin + (lr_mid - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + notch-synchronized growing bursts -> plateau -> climb ---
    # Bursts fire exactly inside the lr notches (repair when steps are small)
    # and vanish for frac >= _F_EXP; the proven bounded endgame then takes over.
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * notch
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + notch dips + fine-hold momentum bump ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_NOTCH) * notch            # dip while repairing
    up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_CENTER) / _B1_UP_WIDTH))
    b1_mid = b1_exp + (_B1_POL - b1_exp) * up                 # heavy ball in the hold
    dn = 1.0 / (1.0 + jnp.exp(-(frac - _B1_DN_CENTER) / _B1_DN_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * dn                   # gated off for the spike

    return lr, alpha, beta1, beta2