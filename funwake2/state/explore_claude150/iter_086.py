import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phase-burst best (+0.0533%): an ACCELERATING
# (period-SHRINKING) restart train realized by a log-time warp, plus an
# epsilon-constraint CONTRACTING VIOLATION BAND (prior-art §7.9, still
# untried). The best splits exploration into three EQUAL cycles; here cycle
# lengths shrink geometrically (~50% / 31% / 19% of exploration), which
# directly implements the parent hint "higher/LONGER lr peak early":
#
#   lr    — 3% warmup, then three cosine restarts under the warp. The FIRST
#           cycle is the longest AND hottest ever tried (peak 1.75*D held
#           over ~half the exploration phase — maximum basin discovery while
#           beta2=0.2 keeps updates sign-like and bounded). Later cycles are
#           shorter and cooler (per-cycle peak decay in warped time down to
#           the proven 1.05*D), so restoration bursts arrive FASTER exactly
#           as feasibility starts to matter. Troughs stay at the proven
#           0.65*D. From 62% the proven straight linear tail lands exactly on
#           gamma_min at the last step.
#   alpha — the proven anti-phased burst train (bursts at lr troughs, growing
#           3 -> 8*alpha0) on top of a RISING exploration floor 0.25 ->
#           1.0*alpha0: early on the tolerance band is wider than any parent
#           (freer AEP/violation trading under the long hot peak), and it
#           contracts open-loop so the layout enters the polish phase with
#           less debt than the constant-0.4 floor allowed. Then the proven
#           endgame verbatim: logistic ramp onto the bounded 6*alpha0 ALM
#           plateau at 66%, cubic-delayed geometric climb from 78% ending at
#           the 5/5-seed-feasible terminal 5*alpha0*D/gamma_min.
#   betas — proven transitions untouched: beta2 0.2 -> 0.9 at the cool-down,
#           beta1 0.1 with the per-burst dip to 0.05 (momentum never carries
#           turbines back across the boundary mid-repair) and the terminal
#           gate to 0.02 during the final alpha spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three restarts inside the exploration phase
_R = 0.62             # cycle-length ratio (<1): each cycle ~62% of the previous
_RN = _R ** 3         # r^N, precomputed as a plain float
_HI0 = 1.75           # hottest, longest first peak yet, in units of D
_HI1 = 1.05           # final peak — the linear tail starts from here (proven)
_LO = 0.65            # bounded trough lr, in units of D (proven)
_A_F0 = 0.25          # exploration alpha floor at the start (widest band yet)
_A_F1 = 1.0           # floor at cool-down: band contracted before the polish
_A_B0 = 3.0           # first restoration burst height, in alpha0 units (proven)
_A_B1 = 8.0           # last restoration burst height, in alpha0 units (proven)
_Q = 3.0              # sharp bursts: alpha stays low most of each cycle (proven)
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp just after cool-down start (proven)
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
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> 3 shrinking-period decaying-peak restarts -> linear tail ---
    # Log-time warp: theta runs 0 -> N over the exploration phase with cycle k
    # occupying a share proportional to r^k, so the hot first cycle is the
    # longest. theta(fc=1) = N exactly, so cos(2*pi*N) = 1 pins the cool-down
    # start at the final (coolest) peak, _HI1 * D — the proven tail property.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    theta = jnp.log(1.0 + fc * (_RN - 1.0)) / jnp.log(_R)     # in [0, N], traceable
    cyc = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * theta))         # 1 at peaks, 0 at troughs
    hi = _HI0 + (_HI1 - _HI0) * (theta / _N_CYC)              # per-cycle peak decay
    lr_cyc = (_LO + (hi - _LO) * cyc) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: contracting floor + anti-phased growing bursts -> plateau -> climb ---
    # burst = 1 exactly at lr troughs, ~0 near lr peaks (and 0 for frac >= _F_COOL,
    # since theta freezes at a peak); the floor rises across exploration so the
    # enforced violation band contracts toward the polish (epsilon-constraint).
    burst = (1.0 - cyc) ** _Q
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    floor = _A_F0 + (_A_F1 - _A_F0) * fc                      # contracting tolerance band
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = floor + (_A_PLAT - floor) * ramp + burst_amp * burst
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