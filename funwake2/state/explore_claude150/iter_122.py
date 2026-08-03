import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-hold best (+0.0577%): the lr waveform is
# inverted from "hold hot, dip briefly" to a METALLURGICAL HEAT-QUENCH-TEMPER
# SAWTOOTH. Each exploration cycle RAMPS lr upward from a warm base to a peak
# HOTTER than anything tried (2.0*D vs the 1.65*D historical max), then
# CRASHES through a smooth quench to a 0.3*D temper hold where a synchronized
# alpha burst repairs constraint debt, then restarts (SGDR-style jump) at the
# next, cooler cycle. The hypothesis: heat applied as a rising ramp lets the
# layout escape progressively larger basins right before each repair locks
# gains in, instead of stewing at one fixed hold temperature.
#
# The LAST cycle deliberately has NO quench — it ramps to 1.15*D and holds,
# so the proven linear cool-down launches hot and continuously from that
# level, and the proven terminal alpha machinery performs the final repair.
#
#   lr    — 3% linear warmup (proven) -> three ascending sawteeth: base
#           0.75*peak -> peak (2.0*D, 1.575*D, 1.15*D), smoothstep quench to
#           0.3*D + temper hold on cycles 1-2 only -> exploration ends at the
#           proven 62% at 1.15*D -> proven straight linear tail landing
#           exactly on gamma_min at the last step.
#   alpha — proven architecture intact: 0.4*alpha0 exploration floor, growing
#           restoration bursts (3 -> 8 alpha0) fired inside the quench/temper
#           windows, logistic ramp to the bounded 6*alpha0 ALM plateau at 66%,
#           and the 5/5-seed-feasible cubic-delayed geometric climb from 78%
#           to the terminal 5*alpha0*D/gamma_min spike.
#   betas — proven transitions: beta2 0.2 -> 0.9 at cool-down; beta1 0.1 with
#           a dip to 0.05 inside each quench window (momentum must not carry
#           turbines back across the boundary mid-repair) and the gated drop
#           to 0.02 during the terminal alpha spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three heat-quench-temper cycles (last one quench-free)
_R1 = 0.70            # in-cycle fraction spent ramping base -> peak
_R2 = 0.85            # quench ends here; temper hold at _LO until cycle end
_PK0 = 2.0            # cycle-1 peak, in units of D — hotter than any prior try
_PK1 = 1.15           # cycle-3 peak; the linear tail launches from here
_BASE_FRAC = 0.75     # each sawtooth starts at 0.75x its own peak
_LO = 0.30            # quench/temper floor lr, in units of D
_A_LO = 0.4           # exploration penalty floor at full heat, in alpha0 units
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
_B1_NOTCH = 0.05      # reduced momentum inside each quench/temper window
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- cycle machinery: fc freezes at 1 past _F_COOL; the frozen state is
    # cycle 3 at t=1, i.e. lr held at _PK1*D with the burst envelope at 0, so
    # the tail launches clean and no burst leaks past the cool-down.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    u = fc * _N_CYC
    kf = jnp.clip(jnp.floor(u), 0.0, _N_CYC - 1.0)            # cycle index 0,1,2
    t = u - kf                                                # in-cycle phase [0,1]
    g = kf / (_N_CYC - 1.0)                                   # 0 -> 1 across cycles
    peak = _PK0 + (_PK1 - _PK0) * g                           # ascending-saw peaks decay per cycle
    base = _BASE_FRAC * peak
    qmask = jnp.where(kf < _N_CYC - 1.5, 1.0, 0.0)            # quench cycles 1-2 only

    # --- lr: rising sawtooth -> smoothstep quench -> temper hold -> tail ---
    rp = jnp.clip(t / _R1, 0.0, 1.0)                          # linear heat ramp
    w = jnp.clip((t - _R1) / (_R2 - _R1), 0.0, 1.0)
    qs = w * w * (3.0 - 2.0 * w)                              # smoothstep crash to _LO
    lr_expl = (base + (peak - base) * rp - (peak - _LO) * qmask * qs) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_expl - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + quench-synchronized growing bursts -> plateau -> climb ---
    # Burst envelope spans quench + temper hold (repair while steps are small),
    # is zero on the quench-free last cycle, and vanishes for frac >= _F_COOL;
    # the proven bounded endgame then takes over.
    benv = qmask * jnp.sin(jnp.pi * jnp.clip((t - _R1) / (1.0 - _R1), 0.0, 1.0)) ** 2
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * benv
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-quench beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_NOTCH) * benv             # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2