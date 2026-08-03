import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the cycle GEOMETRY
# itself. Every lineage schedule — including the best — used EQUAL-LENGTH
# restart cycles. This is reverse-SGDR: three cycles whose lengths CONTRACT
# geometrically (4 : 2 : 1, T_mult = 1/2), realized with a smooth log2-warped
# cycle clock so it stays fully traceable. The change does two things the
# equal-cycle best cannot:
#   (1) it grants the guidance's "higher AND longer first peak" literally —
#       the hottest cycle (peak 1.80*D, above the tried 1.65*D) now owns 4/7
#       of the exploration window, so big basin hops get both the heat and
#       the time budget;
#   (2) restoration bursts arrive at an ACCELERATING cadence (~21%, ~46%,
#       ~58% of the run) with growing strength, so constraint repair
#       naturally densifies toward continuous enforcement just as the proven
#       plateau/terminal endgame takes over — the layout enters the polish
#       phase nearly feasible, exactly the property that made the parent
#       5/5-seed feasible.
# Later bursts are shorter in real time (short cycles), so the burst exponent
# is softened (3 -> 2.5, wider bursts) and the final burst height raised
# (8 -> 9 alpha0) to keep per-burst repair impulse comparable. EVERYTHING
# ELSE — warmup, 62% cool-down split, linear lr tail landing on gamma_min,
# 0.65*D troughs, 1.05*D final peak, alpha floor/logistic ramp/6*alpha0
# plateau/cubic terminal climb to 5*alpha0*D/gamma_min, and all beta
# transitions incl. the per-burst beta1 dip — is frozen from the feasible
# best, making this a clean ablation of cycle contraction + a hotter start.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three restarts, geometrically contracting (4:2:1)
_W = 7.0              # 2**_N_CYC - 1: total cycle-length units
_HI0 = 1.80           # first (long) restart peak — hotter than the tried 1.65
_HI1 = 1.05           # last peak; the linear tail starts from here (proven)
_LO = 0.65            # bounded trough lr, in units of D (proven)
_A_LO = 0.4           # exploration penalty floor at lr peaks, in alpha0 units
_A_B0 = 3.0           # first restoration burst height, in alpha0 units
_A_B1 = 9.0           # last burst height — raised since late bursts are brief
_Q = 2.5              # slightly wider bursts to offset shrinking cycle length
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
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> 3 CONTRACTING cosine restarts -> linear tail ---
    # Warped cycle clock: u hits integers exactly at cycle boundaries
    # fc = 0, 4/7, 6/7, 1, so cycle lengths halve (4:2:1). The log2 argument
    # is >= 1 everywhere, u is smooth, and no Python branching is needed.
    # At fc = 1, u = _N_CYC exactly, so cos(2*pi*u) = 1 pins the cool-down
    # start at the final (coolest) peak, _HI1 * D — same handoff as the best.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    u = _N_CYC - jnp.log2(_W * (1.0 - fc) + 1.0)
    cyc = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * u))   # 1 at cycle starts/ends, 0 at troughs
    hi = _HI0 + (_HI1 - _HI0) * fc                  # decaying peak envelope
    lr_cyc = (_LO + (hi - _LO) * cyc) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)     # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)         # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + accelerating anti-phased bursts -> plateau -> terminal climb ---
    # burst = 1 exactly at lr troughs (~21%, ~46%, ~58% of the run) and ~0 near
    # lr peaks; frozen at 0 past _F_COOL since fc freezes at a peak. Repair
    # therefore always coincides with a near-minimal step size, and the burst
    # cadence densifies into the continuous plateau enforcement.
    burst = (1.0 - cyc) ** _Q
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc        # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * burst
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)  # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-burst beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_BURST) * burst  # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2