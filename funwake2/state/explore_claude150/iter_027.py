import jax.numpy as jnp

# REGIME SHIFT vs the 3-restart anti-phased best (+0.0533%): the exploration
# phase moves from the RESTART regime (3 long cosine cycles) to the FAST
# CYCLICAL-LR / PERTURBATION regime (Smith CLR, prior-art §2/§6 read in its
# short-cycle limit) — 8 short cosine cycles instead of 3 long ones. The
# mechanism is qualitatively different, not a constant tweak:
#   * each cycle is a ~600-step KICK-AND-REPAIR unit, not a full restart, so
#     violation debt is repaid every ~600 steps instead of every ~1650;
#   * that 4x-shorter repair window is what LICENSES a hotter envelope than
#     anything in the lineage (first peak 1.9*D vs the tried 1.65*D ceiling)
#     — the run gets more total time above 1.3*D than any parent while never
#     carrying boundary/spacing debt for long;
#   * troughs sit LOWER (0.55*D vs 0.65*D) so every repair pulse spends its
#     small steps purely pulling turbines inside, then the next hot peak
#     re-explores from a near-feasible layout.
# NEW generalization guard (first use of min_spacing anywhere in the lineage):
# the hot peak is capped at 0.9*min_spacing, so on farms with tight spacing a
# single exploration step can never blow straight through a spacing pair;
# inactive on farms where min_spacing >> D.
# The PROVEN endgame is preserved verbatim: linear lr tail landing exactly on
# gamma_min, logistic alpha ramp to the bounded 6*alpha0 ALM plateau, the
# cubic-delayed geometric climb to the 5/5-seed-feasible terminal
# 5*alpha0*D/gamma_min, beta2 0.2 -> 0.9 at cool-down, and beta1 gated down
# to 0.02 through the terminal spike (with the per-pulse 0.05 dip so momentum
# never fights a repair).
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.60        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 8.0          # EIGHT short kick-and-repair cycles (fast-CLR regime)
_HI0 = 1.90           # first peak — hotter than the lineage's 1.65*D ceiling
_HI1 = 1.00           # final peak; the proven linear tail starts from here
_LO = 0.55            # trough lr in units of D — cooler repairs than 0.65
_MS_CAP = 0.9         # peak lr never exceeds 0.9 * min_spacing (new guard)
_A_LO = 0.35          # alpha floor at hot peaks, in alpha0 units (free ascent)
_A_B0 = 2.0           # first repair-pulse height, in alpha0 units
_A_B1 = 9.0           # last repair-pulse height, in alpha0 units
_Q = 4.0              # sharpens pulses: short cycles must stay mostly low-alpha
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
_B1_PULSE = 0.05      # reduced momentum inside each repair pulse
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    ms = jnp.asarray(min_spacing) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> 8 fast kick-and-repair cosine cycles -> linear tail ---
    # fc freezes at 1 past _F_COOL; with an integer cycle count cos(2*pi*N)=1,
    # so the cool-down starts exactly from the final (coolest) peak, _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    cyc = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * _N_CYC * fc))   # 1 at peaks, 0 at troughs
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying peak envelope
    hi_lr = jnp.maximum(jnp.minimum(hi * Dj, _MS_CAP * ms), _LO * Dj)  # spacing-aware cap
    lr_cyc = _LO * Dj + (hi_lr - _LO * Dj) * cyc
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + fast anti-phased repair pulses -> plateau -> climb ---
    # pulse = 1 exactly at lr troughs and ~0 near peaks (and 0 for
    # frac >= _F_COOL, since fc freezes at a peak), so every repair happens at
    # minimum step size; pulse height grows 2 -> 9 alpha0 across the cycles so
    # the layout enters the polish phase already near-feasible.
    pulse = (1.0 - cyc) ** _Q
    pulse_amp = _A_B0 + (_A_B1 - _A_B0) * fc
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + pulse_amp * pulse
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-pulse beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_PULSE) * pulse            # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2