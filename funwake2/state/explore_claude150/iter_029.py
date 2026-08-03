import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the exploration
# phase is rebuilt as TRUE SGDR BASIN-HOPPING — hard sawtooth restarts with
# T_mult-LENGTHENING cycles and DEEP anneals — instead of the parent's smooth,
# equal-period cosine oscillation. Every prior attempt oscillated lr
# symmetrically (up and down at the same rate), so the layout never fully
# settles; here each cycle is an asymmetric anneal-repair-perturb unit:
#
#   anneal  — cosine decay from a hot peak all the way down to 0.30*D (far
#             below the tried 0.65*D trough floor), so each cycle actually
#             CAPTURES a basin rather than skimming over it;
#   repair  — an end-of-cycle restoration spike (phi^4, growing 3 -> 8 alpha0
#             across cycles) concentrated where lr is smallest, so the
#             incumbent entering every restart is nearly feasible;
#   perturb — an instantaneous jump to the next (cooler) peak, the literal
#             SGDR restart no parent has dared: peaks 1.70*D -> 1.40*D ->
#             1.10*D, hotter than the tried 1.65*D ceiling.
#
# Cycle periods lengthen geometrically (T_mult = 1.6): a short hot scramble
# first, then progressively longer, cooler, deeper anneals — the classic SGDR
# T_mult structure absent from the whole lineage. At 62% a final restart to
# 1.05*D hands over to the PROVEN endgame, kept bit-for-bit: linear lr tail
# landing exactly on gamma_min, logistic alpha ramp to the bounded 6*alpha0
# ALM plateau, cubic-delayed geometric climb to the 5/5-seed-feasible terminal
# 5*alpha0*D/gamma_min, beta2 0.2 -> 0.9 at cool-down, beta1 gated to 0.02 in
# the terminal spike, and the beta1 dip to 0.05 while each repair is pulling
# turbines back inside the polygon.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; polish tail to gamma_min at 100%
_N_CYC = 3            # three hard-restart cycles inside the exploration phase
_T_MULT = 1.6         # geometric cycle-length growth (SGDR T_mult)
_HI0 = 1.70           # first restart peak — hotter than anything tried
_HI1 = 1.10           # last restart peak (peaks decay linearly in k)
_LO = 0.30            # deep anneal floor, in units of D — true basin capture
_MID = 1.05           # final restart level; the proven linear tail starts here
_A_LO = 0.4           # exploration penalty floor, in alpha0 units (proven)
_A_S0 = 3.0           # first end-of-cycle repair spike height, in alpha0 units
_A_S1 = 8.0           # last repair spike height, in alpha0 units
_Q = 4.0              # concentrates each spike at the cycle's deepest anneal
_F_FADE = 0.63        # repair spikes hand over to the plateau ramp here
_W_FADE = 0.01
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
_B1_SPIKE = 0.05      # reduced momentum inside each repair spike
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03

_R_TOT = _T_MULT ** _N_CYC - 1.0   # geometric normalizer for cycle boundaries
_LOG_R = jnp.log(_T_MULT)


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- cycle geometry: N sawtooth cycles with T_mult-lengthening periods ---
    # Boundaries sit at b_k = (T_mult^k - 1)/(T_mult^N - 1); the cycle index k
    # and within-cycle phase phi are recovered branchlessly via log/floor.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    k = jnp.clip(jnp.floor(jnp.log(1.0 + fc * _R_TOT) / _LOG_R), 0.0, _N_CYC - 1.0)
    b0 = (_T_MULT ** k - 1.0) / _R_TOT
    b1 = (_T_MULT ** (k + 1.0) - 1.0) / _R_TOT
    phi = jnp.clip((fc - b0) / (b1 - b0), 0.0, 1.0)           # 0 at restart, 1 at anneal end

    # --- lr: warmup -> hard-restart deep anneals -> final restart + linear tail ---
    hi = _HI0 + (_HI1 - _HI0) * (k / (_N_CYC - 1.0))          # per-cycle decaying peaks
    lr_cycle = (_LO + (hi - _LO) * 0.5 * (1.0 + jnp.cos(jnp.pi * phi))) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_tail = gmin + (_MID * Dj - gmin) * (1.0 - p)           # exact landing on gamma_min
    lr_env = jnp.where(frac < _F_COOL, lr_cycle, lr_tail)     # the final SGDR jump
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + end-of-cycle repair spikes -> plateau -> terminal climb ---
    # spike peaks exactly at each cycle's deepest anneal (phi=1), just before
    # the restart perturbs the layout, so every hop launches from a nearly
    # feasible incumbent; fade retires the spikes as the plateau ramp takes over.
    spike = phi ** _Q
    amp = _A_S0 + (_A_S1 - _A_S0) * fc                        # repairs strengthen per cycle
    fade = 1.0 / (1.0 + jnp.exp((frac - _F_FADE) / _W_FADE))
    repair = spike * fade
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + amp * repair
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-repair beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_SPIKE) * repair           # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2