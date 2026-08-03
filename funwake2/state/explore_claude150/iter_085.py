import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the prior-art
# menu's top UNTRIED lr bet (§6 WSD / §2 one-cycle) — warmup -> long STABLE
# HOT HOLD -> single linear cool-down — replacing the SGDR cosine restarts
# that all eight recent (failed) attempts kept re-tuning. Instead of spending
# most of the exploration budget climbing in and out of cosine valleys, the
# layout explores at full, sustained heat; the proven MID-RUN FEASIBILITY-
# RESTORATION BURSTS are retained, but now cut brief NOTCHES into the hold
# (lr dips to 0.65x while each burst repairs), so restoration still happens
# at reduced step size without sacrificing the hot plateau between repairs.
# Two further firsts for the lineage:
#   * the hold amplitude is floored by MIN_SPACING — the true basin-hop
#     length scale, never used by any parent — so the hot phase can swap
#     neighbouring turbines on any farm geometry, not just when D ~ spacing;
#   * the hold carries a gentle WSD tilt (1.28*D -> 1.10*D) so sustained heat
#     is front-loaded ("slightly higher/longer early", per guidance) while
#     the linear tail still launches from the proven ~1.05-1.10*D scale.
# The entire 5/5-seed-feasible endgame is preserved bit-for-bit: linear lr
# tail landing exactly on gamma_min, logistic alpha ramp to the bounded
# 6*alpha0 ALM plateau, cubic-delayed geometric climb to the terminal
# 5*alpha0*D/gamma_min spike, beta2 0.2 -> 0.9 at cool-down, and beta1 gated
# to 0.02 in the terminal spike with per-burst dips to 0.05.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_B = 3.0            # three restoration bursts inside the exploration phase
_HOLD_D = 1.28        # hot hold in units of D (sustained, vs brief 1.65 peaks)
_HOLD_S = 0.55        # hold floor in units of min_spacing (basin-hop scale)
_TILT = 0.14          # WSD tilt: hold decays 1.28*D -> ~1.10*D across exploration
_NOTCH = 0.35         # lr dips to 0.65x the hold inside each restoration burst
_Q = 3.0              # sharpens bursts so lr/alpha sit at hold/floor most of the time
_A_LO = 0.4           # exploration penalty floor between bursts, in alpha0 units
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
_B1_BURST = 0.05      # reduced momentum inside each restoration burst
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    msj = jnp.asarray(min_spacing) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- burst waveform: 3 sharp bumps at the proven positions, 0 elsewhere ---
    # fc freezes at 1 past _F_COOL, where cos(2*pi*N) = 1 -> burst = 0, so the
    # cool-down launches cleanly from the (tilted) hold value.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    cyc = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * _N_B * fc))     # 1 between bursts, 0 at centres
    burst = (1.0 - cyc) ** _Q                                 # 1 at burst centres

    # --- lr: warmup -> tilted hot hold with burst notches -> linear tail ---
    hold = jnp.maximum(_HOLD_D * Dj, _HOLD_S * msj) * (1.0 - _TILT * fc)
    lr_x = hold * (1.0 - _NOTCH * burst)                      # notch during repairs only
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_x - gmin) * (1.0 - p)                 # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + growing restoration bursts -> plateau -> terminal climb ---
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * burst
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-burst beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_BURST) * burst            # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2