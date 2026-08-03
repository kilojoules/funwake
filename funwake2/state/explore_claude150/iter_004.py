import jax.numpy as jnp

# STRUCTURALLY NEW vs the native-coupled parent: the penalty is DECOUPLED from
# 1/lr for the whole main run. Three-phase design from the prior-art menu:
#   lr    — one-cycle / WSD: warmup -> long hold at D -> half-cosine cool-down
#           landing exactly on gamma_min (no compounding product decay).
#   alpha — delayed logistic ramp (graduated / LANCELOT) from a LOW exploration
#           penalty to a BOUNDED plateau of a few*alpha0 (exact-penalty guidance),
#           then a gated TERMINAL feasibility-restoration divergence that
#           recovers the parent's proven endgame magnitude (~5*alpha0*D/gamma_min),
#           guaranteeing strict feasibility at finish.
#   beta2 — phase transition: native 0.2 while exploring, ramped to 0.9 in the
#           cool-down/feasibility phase (beta2 up, beta1 held low).
_C_PEAK = 1.0          # exploration lr peak = 1.0 * D (hotter + longer than parent)
_F_WARM = 0.02         # linear lr warmup over first 2% of steps
_F_HOLD_END = 0.55     # hold at peak until 55%, then cosine cool-down to gamma_min
_ALPHA_LO = 0.5        # early penalty: half the native start -> freer exploration
_ALPHA_PLATEAU = 6.0   # bounded mid-run plateau, in units of alpha0
_RAMP_CENTER = 0.35    # delayed alpha ramp engages around 35% of the run
_RAMP_WIDTH = 0.06
_GATE_CENTER = 0.90    # terminal restoration engages over the final ~10%
_GATE_WIDTH = 0.02
_TERM_GAIN = 5.0       # terminal alpha ~ 5*alpha0*D/lr_env (parent-proven scale)
_BETA2_LO = 0.2
_BETA2_HI = 0.9        # 10-step memory: stable, still tracks the alpha blow-up
_B2_CENTER = 0.55      # beta2 transition aligned with the lr cool-down start
_B2_WIDTH = 0.05


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = float(int(total_steps))
    lr0 = _C_PEAK * float(D)
    gmin = float(gamma_min)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: one-cycle envelope, hold at lr0 then half-cosine to gamma_min ---
    p = jnp.clip((frac - _F_HOLD_END) / (1.0 - _F_HOLD_END), 0.0, 1.0)
    lr_env = gmin + 0.5 * (lr0 - gmin) * (1.0 + jnp.cos(jnp.pi * p))

    # short linear warmup damps the hot start so the feasible grid init survives
    warm = jnp.minimum(frac / _F_WARM, 1.0)
    lr = lr_env * warm

    # --- alpha: delayed logistic ramp to a bounded plateau (decoupled) ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _RAMP_CENTER) / _RAMP_WIDTH))
    alpha_mid = alpha0 * (_ALPHA_LO + (_ALPHA_PLATEAU - _ALPHA_LO) * ramp)

    # --- terminal feasibility restoration: gated native-style divergence ---
    # alpha_term = gain * alpha0 * D / lr_env -> gain * mean|grad J| / lr_env,
    # diverging as lr_env -> gamma_min, so late feasibility is enforced exactly
    # as strongly as in the feasible parent (5/5 seeds).
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _GATE_CENTER) / _GATE_WIDTH))
    alpha_term = _TERM_GAIN * alpha0 * float(D) / jnp.maximum(lr_env, 1e-30)
    alpha = alpha_mid + gate * alpha_term

    # --- betas: keep beta1 low (native), ramp beta2 up for the polish phase ---
    beta1 = 0.1
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _BETA2_LO + (_BETA2_HI - _BETA2_LO) * b2r

    return lr, alpha, beta1, beta2