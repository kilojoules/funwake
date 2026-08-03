import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the exploration
# lr waveform is changed from cosine SGDR restarts to a DUTY-CYCLED TRAPEZOID
# (WSD-style hot holds, prior-art §6/§2). A cosine restart spends most of each
# cycle in transit between peak and trough; here each cycle HOLDS the hot lr
# for ~75% of its length (square top), then drops into a short, concentrated
# REPAIR NOTCH at a far lower lr (0.40*D vs the 0.65*D cosine troughs). This
# roughly doubles time-at-peak-lr without raising the peak — the "longer hot
# phase" the search guidance asks for, delivered structurally rather than by
# nudging a constant — while the notches keep the parent's proven mid-run
# feasibility-restoration mechanism (filter/funnel, §7.5) intact.
#
#   lr    — 3% linear warmup -> three trapezoid cycles: hold at a decaying
#           hot level (1.65*D -> 1.05*D envelope, the proven peak range) with
#           a ~25%-duty repair notch at 0.40*D per cycle; exploration ends at
#           62% pinned on the final (coolest) hold, then the proven straight
#           linear tail lands exactly on gamma_min at the last step.
#   alpha — anti-phased with the notches: 0.4*alpha0 floor during the hot
#           holds (basin hops trade violation for AEP freely); inside each
#           notch a repair burst rises 3.5 -> 9 alpha0 across cycles (slightly
#           stronger than the parent's 3 -> 8, to repay the extra violation
#           debt the longer holds accrue). After 62% the proven bounded
#           endgame is kept EXACTLY: logistic ramp to the 6*alpha0 ALM
#           plateau, then the cubic-delayed geometric climb from 78% onto the
#           5/5-seed-feasible terminal 5*alpha0*D/gamma_min spike.
#   betas — proven transitions preserved (beta2 0.2 -> 0.9 at cool-down;
#           beta1 gated 0.1 -> 0.02 in the terminal spike), with the per-notch
#           beta1 dip deepened to 0.04 so momentum from the long hot hold is
#           shed the moment a repair notch starts pulling turbines back in.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three hold+notch cycles inside the exploration phase
_HI0 = 1.65           # first hold level, in units of D (proven peak)
_HI1 = 1.05           # last hold level; the linear tail starts from here (proven)
_LO_R = 0.40          # repair-notch lr, in units of D — well below cosine troughs
_C_TH = 0.15          # cycle-shape threshold: notch when cos-cycle < 0.15 (~25% duty)
_W_S = 0.04           # softness of the hold<->notch switch (keeps it traceable-smooth)
_A_LO = 0.4           # exploration penalty floor during holds, in alpha0 units
_A_B0 = 3.5           # first repair-burst height, in alpha0 units
_A_B1 = 9.0           # last repair-burst height, in alpha0 units
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
_B1_HI = 0.1          # native momentum during holds and the polish phase
_B1_REPAIR = 0.04     # deep momentum shed inside each repair notch
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- cycle shape: cosine carrier sharpened into a ~75/25 hold/notch wave ---
    # fc freezes at 1 past _F_COOL, so cyc = 1 (a hold) pins the cool-down
    # start at the final hold level, _HI1 * D, and repair = 0 for the tail.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    cyc = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * _N_CYC * fc))   # 1 mid-hold, 0 mid-notch
    repair = 1.0 / (1.0 + jnp.exp(-(_C_TH - cyc) / _W_S))     # ~1 in notch, ~0 in hold

    # --- lr: warmup -> trapezoid holds with repair notches -> linear tail ---
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying hold envelope
    lr_cyc = (hi * (1.0 - repair) + _LO_R * repair) * Dj      # square top, deep notch
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + growing notch bursts -> plateau -> terminal climb ---
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * repair
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + deepened per-notch beta1 shed ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_REPAIR) * repair          # shed momentum in notches
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2