import jax.numpy as jnp

# STRUCTURAL BREAK from the burst/restart lineage (best +0.0533%). Every
# recent attempt reshuffled cosine restarts + anti-phased bursts and plateaued,
# so this abandons cycles entirely and takes the four prior-art menu bets the
# lineage has never tried, in one coherent design:
#
#   lr    — WSD / hold-then-cool (§6, untried): 4% linear warmup, then a
#           SUSTAINED hot plateau at 1.35*D for 60% of the run. No restarts,
#           no troughs — the time-averaged exploration heat (~1.35*D) exceeds
#           the restart schedules' cycle average (~1.0*D) without exceeding
#           the instantaneous 1.65*D already proven safe. Cool-down is the
#           WSD-canonical 1-sqrt shape: lr falls FAST off the plateau (so
#           repair happens at small step sizes almost immediately) then
#           glides slowly, landing exactly on gamma_min at the last step.
#   alpha — ε-CONSTRAINED CONTRACTING TOLERANCE (§7.9, untried), fully
#           decoupled from lr. During warmup+plateau alpha is an ADMM-style
#           CONSTANT moderate penalty, 1.5*alpha0 (untried: the lineage used
#           either a low floor with bursts or a coupled ramp) — enough to keep
#           violation debt bounded while the plateau trades feasibility for
#           AEP. From 60% the enforced violation band contracts geometrically
#           from ~D-scale to gamma_min: alpha climbs exponentially in the log
#           domain, back-loaded with power 2.5, passing the old 6*alpha0
#           plateau level near 80% and landing EXACTLY on the proven
#           5/5-seed-feasible terminal 5*alpha0*D/gamma_min at the last step.
#           Net: stricter than the parent over the final 15% (where
#           feasibility is decided), looser mid-run (where AEP is decided).
#   beta1 — one-cycle anti-correlation with lr (§2/§4, untried): LOW momentum
#           (0.06) on the hot plateau so 1.35*D steps never compound across
#           the boundary; RISES to 0.12 (Sutskever increasing ramp) once lr
#           collapses, letting momentum act as an implicit ALM multiplier
#           during polish; then the proven terminal gate drops it to 0.02
#           while the alpha spike collects the last of the debt.
#   beta2 — proven transition kept verbatim: 0.2 while exploring, logistic
#           rise to 0.9 aligned with the cool-down start.
#
# Feasibility anchors preserved from the proven endgame: exact gamma_min lr
# landing, terminal alpha scale 5*alpha0*D/gamma_min, beta2 high + beta1 ~0
# during the terminal restoration spike.
_F_WARM = 0.04        # linear lr warmup fraction
_F_STAB = 0.60        # plateau ends here; cool-down + tolerance contraction begin
_LR_HOT = 1.35        # sustained plateau lr, in units of D
_A_EXPL = 1.5         # ADMM-style constant penalty during exploration, in alpha0 units
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven feasible scale)
_POW = 2.5            # back-loading of the geometric tolerance contraction
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 rise aligned with cool-down start
_B2_WIDTH = 0.05
_B1_HOT = 0.06        # low momentum while lr is hot (one-cycle anti-correlation)
_B1_MID = 0.12        # raised momentum during low-lr polish (increasing ramp)
_B1_LO = 0.02         # near-zero momentum inside the terminal alpha spike (proven)
_B1_C1 = 0.60         # momentum rise aligned with cool-down start
_B1_W1 = 0.05
_B1_C2 = 0.88         # proven terminal beta1 gate
_B1_W2 = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> sustained hot plateau -> 1-sqrt cool-down to gamma_min ---
    warm = jnp.minimum(frac / _F_WARM, 1.0)
    p = jnp.clip((frac - _F_STAB) / (1.0 - _F_STAB), 0.0, 1.0)
    cool = 1.0 - jnp.sqrt(p)                 # fast initial drop, slow glide; 0 at end
    lr = (gmin + (_LR_HOT * Dj - gmin) * cool) * warm   # exact landing on gamma_min

    # --- alpha: constant moderate penalty -> geometric tolerance contraction ---
    # Enforced violation band shrinks as (gmin/D)^u; alpha grows inversely,
    # from 1.5*alpha0 on the plateau to exactly 5*alpha0*D/gmin at the end.
    u = p ** _POW
    log_gain = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_EXPL), 1.0))
    alpha = alpha0 * _A_EXPL * jnp.exp(u * log_gain)

    # --- betas: anti-correlated beta1 (hot->low, cool->raised, terminal->gated) ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    r1 = 1.0 / (1.0 + jnp.exp(-(frac - _B1_C1) / _B1_W1))
    r2 = 1.0 / (1.0 + jnp.exp(-(frac - _B1_C2) / _B1_W2))
    b1_mid = _B1_HOT + (_B1_MID - _B1_HOT) * r1
    beta1 = b1_mid + (_B1_LO - b1_mid) * r2

    return lr, alpha, beta1, beta2