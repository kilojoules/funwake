import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-hold best (+0.0577%): the exploration
# waveform is INVERTED from "hold hot, dip briefly to repair" to a SAWTOOTH
# REVERSE-ANNEAL — each cycle STARTS moderate right after a repair, RAMPS
# heat upward (sqrt-shaped: fast climb, long time near peak), reaches its
# HOTTEST steps immediately BEFORE a dedicated end-of-cycle crash-repair,
# then crashes into a deep notch where a synchronized alpha burst fixes the
# damage while steps are small. Three structural changes at once:
#
#   1. WAVEFORM: ascending sawtooth per cycle instead of a flat hold. Peak
#      heat (1.85*D -> 1.25*D) is HIGHER than any tried hold/peak, but it is
#      reached gradually — after each repair the layout first polishes at
#      moderate lr, then progressively escapes, so hot disruption is always
#      immediately followed by repair, never by more disruption.
#   2. REPAIR CADENCE: notches sit at cycle ENDS (frac ~0.21, 0.41, 0.62),
#      so the FINAL full repair coincides exactly with cool-down start — the
#      tail launches from a just-repaired, feasible configuration.
#   3. REPAIR-THEN-REHEAT HANDOFF: after that final repair the tail jumps
#      back up to 1.15*D (slightly hotter launch than the 1.1*D best) and
#      runs the proven straight linear decay onto gamma_min, under the proven
#      bounded ALM plateau — the same lr/alpha regime the 5/5-feasible
#      parents survived.
#
#   alpha — proven endgame kept byte-for-byte: 0.4*alpha0 exploration floor,
#           growing crash-synchronized bursts (3 -> 8 alpha0) that are gated
#           OFF at cool-down, logistic ramp to the bounded 6*alpha0 plateau
#           at 66%, cubic-delayed geometric climb from 78% to the terminal
#           5*alpha0*D/gamma_min feasibility spike.
#   betas — proven transitions: beta2 0.2 -> 0.9 at cool-down; beta1 0.1
#           with a dip to 0.05 inside each crash-repair (momentum must not
#           drag turbines back over the boundary mid-repair) and the gated
#           drop to 0.02 during the terminal alpha spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear tail to gamma_min at 100%
_N_CYC = 3.0          # three ramp-crash cycles inside the exploration phase
_W = 0.18             # crash-repair window: last 18% of each cycle
_RP = 0.5             # sqrt rise: reach ~70% of the range by mid-cycle
_S = 0.55             # cycle-restart lr right after a repair, in units of D
_P0 = 1.85            # first-cycle peak lr, in units of D — hotter than any hold tried
_P1 = 1.25            # last-cycle peak lr, in units of D
_LO = 0.3             # crash bottom — deep, surgical repair steps
_LAUNCH = 1.15        # tail launch lr after the final repair, in units of D
_A_LO = 0.4           # exploration penalty floor, in alpha0 units (proven)
_A_B0 = 3.0           # first repair-burst height, in alpha0 units (proven)
_A_B1 = 8.0           # last repair-burst height, in alpha0 units (proven)
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_A_GATE_W = 0.01      # sharp gate killing exploration bursts once the tail starts
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while ramping and polishing
_B1_NOTCH = 0.05      # reduced momentum inside each crash-repair
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr, exploration: ascending sawtooth with end-of-cycle crash-repair ---
    # u in [0,1) is the phase within the current cycle. The ramp occupies
    # u in [0, 1-_W); the crash-repair descent occupies the final _W of the cycle.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    u = jnp.mod(fc * _N_CYC, 1.0)
    ur = jnp.clip(u / (1.0 - _W), 0.0, 1.0)
    rise = ur ** _RP                                          # sqrt-shaped heat build
    peak = _P0 + (_P1 - _P0) * fc                             # decaying peak envelope
    lr_ramp = _S + (peak - _S) * rise                         # units of D
    un = jnp.clip((u - (1.0 - _W)) / _W, 0.0, 1.0)
    m = un * un * (3.0 - 2.0 * un)                            # smoothstep crash, 0 -> 1
    lr_exp = (lr_ramp * (1.0 - m) + _LO * m) * Dj

    # --- lr, tail: repair-then-reheat launch, straight line onto gamma_min ---
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_tail = gmin + (_LAUNCH * Dj - gmin) * (1.0 - p)        # exact landing on gamma_min
    lr = jnp.where(frac < _F_COOL, lr_exp, lr_tail)
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the start; lr only
    lr = lr * warm

    # --- alpha: floor + crash-synchronized growing bursts -> plateau -> climb ---
    # Bursts track the crash descent m (repair exactly while steps shrink) and
    # are sharply gated off once the tail starts; the proven endgame takes over.
    gate = 1.0 / (1.0 + jnp.exp((frac - _F_COOL) / _A_GATE_W))
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * m * gate
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-crash beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_NOTCH) * (m * gate)       # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2