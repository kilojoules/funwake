import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-notch best (+0.0577%): the schedule no
# longer modulates one optimizer — it ALTERNATES BETWEEN TWO OPTIMIZER
# IDENTITIES. Every prior attempt (cosine SGDR, phased bursts, WSD hold)
# kept the moments essentially native (beta1~0.1, beta2 0.2->0.9) and varied
# only lr/alpha waveforms; the moment axis of the prior-art menu — "standard
# Adam (0.9, 0.999) vs TopFarm (0.1, 0.2)" and highest-value bet #4
# ("phase-transition the Adam moments WITH the alpha phase") — is untried.
#
# Here the proven repair gate (three narrow sin^12 windows, ~20% duty) now
# switches the ENTIRE optimizer character, not just lr depth:
#   EXPLORE blocks (~80% of each cycle): a standard-Adam explorer
#     (beta1=0.9, beta2=0.999) at the proven hot hold (1.5*D -> 1.1*D).
#     Long-memory beta2 equalizes per-turbine step sizes, so weak-gradient
#     downstream turbines migrate as far as upstream ones, and beta1=0.9
#     gives directional persistence to coast through shallow wake minima —
#     a qualitatively different search dynamic at the same lr heat.
#   REPAIR blocks (gate~1): a TopFarm-like repairer (beta1=0.05, beta2=0.3,
#     lr=0.35*D) under the proven growing alpha bursts (3 -> 8 alpha0):
#     short-memory moments react instantly to constraint gradients and
#     momentum cannot carry turbines back across the boundary mid-repair.
#
# Everything downstream of exploration is the PROVEN 5/5-feasible endgame,
# untouched: exploration ends at 62%, straight linear lr tail landing exactly
# on gamma_min; logistic alpha ramp to the bounded 6*alpha0 ALM plateau at
# 66%; cubic-delayed geometric climb from 78% to the 5*alpha0*D/gamma_min
# terminal feasibility spike; a single logistic "identity handoff" at 64%
# retires the explorer into the proven polish moments (beta1=0.1, beta2=0.9),
# with the gated beta1 drop to 0.02 during the terminal spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three repair windows inside the exploration phase
_Q = 6.0              # gate = sin^(2Q); ~80% of each cycle in explore identity
_HI0 = 1.5            # initial hold level, in units of D (proven hot envelope)
_HI1 = 1.1            # final hold level; the linear tail starts from here
_LO = 0.35            # repair-window lr, in units of D (proven)
_A_LO = 0.4           # exploration penalty floor, in alpha0 units (proven)
_A_B0 = 3.0           # first repair burst height, in alpha0 units (proven)
_A_B1 = 8.0           # last repair burst height, in alpha0 units (proven)
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B1_EXPLORE = 0.9     # standard-Adam explorer identity
_B2_EXPLORE = 0.999
_B1_REPAIR = 0.05     # short-memory repairer identity
_B2_REPAIR = 0.3
_B1_POLISH = 0.1      # proven polish moments after the identity handoff
_B2_POLISH = 0.9
_ID_CENTER = 0.64     # explorer -> polisher handoff, just after cool-down start
_ID_WIDTH = 0.03
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- identity gate: 0 = explorer, ~1 = repairer; frozen at 0 past cool-down ---
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    gate = jnp.sin(jnp.pi * _N_CYC * fc) ** (2.0 * _Q)

    # --- lr: warmup -> hot hold with repair dips -> proven linear tail ---
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying hold envelope
    lr_hold = (_LO + (hi - _LO) * (1.0 - gate)) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + gate-synchronized growing bursts -> plateau -> climb ---
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * gate
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: explorer <-> repairer alternation, then handoff to polisher ---
    b1_cyc = _B1_EXPLORE + (_B1_REPAIR - _B1_EXPLORE) * gate
    b2_cyc = _B2_EXPLORE + (_B2_REPAIR - _B2_EXPLORE) * gate
    handoff = 1.0 / (1.0 + jnp.exp(-(frac - _ID_CENTER) / _ID_WIDTH))
    b1_mid = b1_cyc + (_B1_POLISH - b1_cyc) * handoff
    beta2 = b2_cyc + (_B2_POLISH - b2_cyc) * handoff
    b1_term = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * b1_term

    return lr, alpha, beta1, beta2