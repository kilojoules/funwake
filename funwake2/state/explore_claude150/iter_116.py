import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-notch best (+0.0577%): the REPAIR CADENCE
# is CHIRPED instead of periodic, and the exploration penalty floor is
# GRADUATED instead of constant. The parent spaces its three repair notches
# evenly, so the very first hot span — the only time the layout is disordered
# enough for basin hopping to pay — is cut short at ~10% of the run, while the
# late exploration phase gets the same sparse repair rate even though debt
# accumulated there has little time left to be repaid gracefully.
#
# Here the notch phase advances as fc^1.7: ONE long uninterrupted hot span
# covers the first ~18% of the run (nearly triple the parent's, at a hotter
# 1.7*D hold), then four repair notches arrive at an ACCELERATING cadence
# (~18%, 35%, 47%, 57% of the run), so constraint debt is serviced more and
# more frequently as the endgame approaches — graduated alternation between
# an almost-unconstrained AEP phase and surgical repair phases. The alpha
# floor is likewise graduated (0.2 -> 0.55 alpha0 across exploration): early
# steps climb nearly pure AEP (delayed-ramp / graduated-penalty, prior-art
# §7.3), with the integrated penalty budget kept near the parent's proven
# level by back-loading the floor. The entire proven feasibility endgame —
# bounded ALM plateau, cubic-delayed terminal spike, lr tail landing exactly
# on gamma_min, beta gating — is preserved verbatim (5/5-seed machinery).
#
#   lr    — 3% linear warmup (proven) -> flat-top hold, envelope 1.7*D ->
#           1.1*D, with four CHIRPED sin^12 notches down to 0.35*D ->
#           exploration ends at the proven 62% -> proven straight linear tail
#           landing exactly on gamma_min at the last step.
#   alpha — graduated exploration floor 0.2 -> 0.55 alpha0, growing
#           restoration bursts (3 -> 8 alpha0) synchronized with the chirped
#           notches, logistic ramp to the bounded 6*alpha0 ALM plateau at
#           66%, and the proven cubic-delayed geometric climb from 78% to the
#           terminal 5*alpha0*D/gamma_min spike.
#   betas — proven transitions: beta2 0.2 -> 0.9 at cool-down; beta1 0.1 with
#           a dip to 0.05 inside each (chirped) repair notch and the gated
#           drop to 0.02 during the terminal alpha spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 4.0          # four repair notches inside the exploration phase
_CHIRP = 1.7          # phase exponent: sparse repairs early, frequent late
_Q = 6.0              # notch = sin^(2Q); narrow, square-ish repair windows
_HI0 = 1.7            # hotter initial hold (units of D) — the long first span earns it
_HI1 = 1.1            # final hold level; the linear tail starts from here
_LO = 0.35            # notch-bottom lr — deep, surgical repair windows (proven)
_A_FLOOR0 = 0.2       # near-free early exploration (alpha0 units)
_A_FLOOR1 = 0.55      # graduated floor by end of exploration; mean ~ parent's 0.4
_A_B0 = 3.0           # first restoration burst height, in alpha0 units (proven)
_A_B1 = 8.0           # last restoration burst height, in alpha0 units (proven)
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
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_NOTCH = 0.05      # reduced momentum inside each repair notch
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> flat-top hold with four CHIRPED deep notches -> tail ---
    # phase = pi*N*fc^CHIRP: sin is 0 at fc=0 and fc=1 (clean launch into the
    # tail from the hold level _HI1*D); notch centers land at fc where the
    # phase hits (m-1/2)*pi — increasingly dense as fc -> 1. fc freezes at 1
    # past _F_COOL, where the notch is exactly closed.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    phase = jnp.pi * _N_CYC * fc ** _CHIRP
    notch = jnp.sin(phase) ** (2.0 * _Q)
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying hold envelope
    lr_hold = (_LO + (hi - _LO) * (1.0 - notch)) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: graduated floor + chirp-synchronized growing bursts ->
    # --- plateau -> terminal climb ---
    # Bursts fire exactly inside the lr notches (repair when steps are small)
    # and vanish for frac >= _F_COOL; the proven bounded endgame then takes over.
    floor = _A_FLOOR0 + (_A_FLOOR1 - _A_FLOOR0) * fc          # delayed-ramp exploration floor
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = floor + (_A_PLAT - floor) * ramp + burst_amp * notch
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