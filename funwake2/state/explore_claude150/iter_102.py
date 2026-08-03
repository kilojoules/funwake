import jax.numpy as jnp

# REGIME SHIFT vs both proven exploration regimes: the lineage has tried a
# CONSTANT-period slow-restart schedule (3 long cycles, the +0.0533% best)
# and a CONSTANT-period fast kick-and-repair schedule (8 short cycles, the
# +0.0467% parent). This schedule is a CHIRPED (frequency-swept) CLR — the
# cycle period is not a constant at all but sweeps coarse -> fine within one
# run (phase = 2*pi*N*fc^2, so instantaneous frequency grows linearly in
# exploration time). Mechanism, not a constant tweak:
#   * the FIRST cycle is longer and hotter than the best's first restart
#     (~1390 steps near a 1.7*D envelope vs ~830 near 1.65*D) — exactly the
#     "higher/LONGER early peak" the search state asks for, granted while the
#     layout is still plastic and a deep basin hop is cheapest;
#   * successive cycles contract (~1390 -> ~430 steps), so by late
#     exploration the run is in the parent's proven fast kick-and-repair
#     regime: violation debt is repaid every few hundred steps and the polish
#     phase inherits a near-feasible layout;
#   * alpha bursts stay ANTI-PHASED with the lr troughs (filter/funnel
#     restoration, prior-art §7.5) and — because burst width is fixed in
#     PHASE — early repairs are automatically long-and-gentle while late
#     repairs are short-and-strict, an anneal the constant-period parents
#     cannot express.
# Generalization guard kept: peak lr is capped at 0.9*min_spacing so a single
# hot step can never jump a spacing pair on tight farms.
# The PROVEN endgame is preserved verbatim: linear lr tail landing exactly on
# gamma_min, logistic alpha ramp to the bounded 6*alpha0 ALM plateau, the
# cubic-delayed geometric climb to the 5/5-seed-feasible terminal
# 5*alpha0*D/gamma_min, beta2 0.2 -> 0.9 at cool-down, and beta1 gated to
# 0.02 through the terminal spike with the per-repair 0.05 momentum dip.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.60        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 6.0          # total chirped cycles across the exploration phase
_KAPPA = 2.0          # chirp exponent: phase progress = fc^2 (coarse -> fine)
_HI0 = 1.70           # first-cycle peak envelope (between 1.65 slow / 1.9 fast)
_HI1 = 1.00           # final peak; the proven linear tail starts from here
_LO = 0.60            # trough lr in units of D
_MS_CAP = 0.9         # peak lr never exceeds 0.9 * min_spacing (kept guard)
_A_LO = 0.35          # alpha floor at hot peaks, in alpha0 units (free ascent)
_A_B0 = 2.5           # first repair-burst height, in alpha0 units
_A_B1 = 9.0           # last repair-burst height, in alpha0 units
_Q = 4.0              # sharpens bursts: alpha stays low for most of each cycle
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.64      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_PULSE = 0.05      # reduced momentum inside each repair burst
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    ms = jnp.asarray(min_spacing) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> chirped coarse-to-fine cosine cycles -> linear tail ---
    # prog = fc^2 is the chirp's phase progress: cycle k ends at fc=sqrt(k/N),
    # so early cycles are long (deep hot restarts) and late ones short
    # (kick-and-repair). prog freezes at 1 past _F_COOL and cos(2*pi*N)=1, so
    # the cool-down starts exactly from the final (coolest) peak, _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    prog = fc ** _KAPPA
    cyc = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * _N_CYC * prog))  # 1 at peaks, 0 at troughs
    hi = _HI0 + (_HI1 - _HI0) * prog          # envelope decays with phase progress:
    #                                           the long first cycle stays near-hot
    hi_lr = jnp.maximum(jnp.minimum(hi * Dj, _MS_CAP * ms), _LO * Dj)  # spacing cap
    lr_cyc = _LO * Dj + (hi_lr - _LO * Dj) * cyc
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + anti-phased chirped repair bursts -> plateau -> climb ---
    # pulse = 1 exactly at lr troughs and ~0 near peaks (0 past _F_COOL since
    # prog freezes at a peak). Burst height grows with TIME (fc), not phase, so
    # the first repair after the long hot restart is already firm (~4.4*alpha0)
    # and the last pre-polish repair is near-strict (~8.7*alpha0).
    pulse = (1.0 - cyc) ** _Q
    pulse_amp = _A_B0 + (_A_B1 - _A_B0) * fc
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + pulse_amp * pulse
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-burst beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_PULSE) * pulse            # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2