import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phase-burst best (+0.0533%): NO cycles, NO
# restarts, NO bursts. This is the two menu directions the lineage has never
# combined — an ADMM-STYLE CONSTANT MODERATE PENALTY through exploration and
# an EPSILON-CONSTRAINED SHRINKING TOLERANCE BAND (prior-art §7.9) realized as
# a single piecewise-GEOMETRIC alpha climb — riding on the prior-art §2/§6
# ONE-CYCLE / WSD lr: warmup -> slanted hot hold -> straight linear tail.
#
#   lr    — one-cycle trapezoid instead of cosine restarts: 3% linear warmup
#           to the hottest sustained level any attempt has run (1.55*D), a
#           slanted hold decaying gently to 1.05*D at 55% (far MORE total
#           exploration heat than the restart trains, whose troughs spent
#           half the phase at 0.55-0.65*D), then the proven WSD linear tail
#           landing exactly on gamma_min at the last step. "Hold near c*D,
#           then linear cool-down beats cosine" is the §6 bet, untested here.
#   alpha — fully decoupled from lr and MONOTONE: log-space interpolation
#           through fixed knots, i.e. piecewise-geometric contraction of the
#           enforced violation band. Near-constant moderate 0.7->1.0*alpha0
#           over the whole hot hold (ADMM-style: debt stays bounded without
#           ever paying a burst), a delayed geometric ramp onto the proven
#           6*alpha0 ALM scale as the tail begins, a gentle 9->20*alpha0
#           polish rise, and a steep terminal geometric climb ending at the
#           proven 5*alpha0*D/gamma_min — the same terminal restoration that
#           made the lineage 5/5-seed feasible, reached as the band's final
#           contraction onto gamma_min rather than a bolted-on spike.
#   betas — beta2 0.2 -> 0.9 at the cool-down start (proven). beta1 runs the
#           §2 one-cycle ANTI-CORRELATION: native 0.1 while lr is hot, rising
#           to 0.22 as lr cools (momentum as implicit ALM multiplier, menu
#           bet 4), then the proven gate to 0.02 for the terminal climb so
#           the diverging alpha never rides momentum.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.55        # slanted hold ends here; linear tail to gamma_min at 100%
_LR_PEAK = 1.55       # hold entry, in units of D — hottest sustained lr tried
_LR_END = 1.05        # hold exit; the proven WSD tail starts from here
_A_X = (0.0, 0.50, 0.70, 0.82, 0.92, 1.0)   # alpha knot positions (frac)
_A_Y = (0.7, 1.0, 6.0, 9.0, 20.0)           # alpha knots, in alpha0 units
_TERM_GAIN = 5.0      # final knot = 5*alpha0*D/gamma_min (proven terminal scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_BASE = 0.1        # native momentum during the hot hold
_B1_POLISH = 0.22     # one-cycle rise: more momentum once lr has cooled
_B1_RC = 0.62         # center of the momentum rise (just after cool-down starts)
_B1_RW = 0.04
_B1_LO = 0.02         # near-zero momentum during the terminal alpha climb (proven)
_B1_GC = 0.88
_B1_GW = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)
    one = jnp.ones_like(Dj)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: one-cycle trapezoid — warmup -> slanted hot hold -> WSD tail ---
    # Piecewise-linear in frac via interp; the last knot is gamma_min itself,
    # so the tail lands exactly on gamma_min at the final step.
    lr_x = jnp.asarray([0.0, _F_WARM, _F_COOL, 1.0])
    lr_y = jnp.stack([0.0 * one, _LR_PEAK * one, _LR_END * one, gmin / Dj])
    lr = Dj * jnp.interp(frac, lr_x, lr_y)

    # --- alpha: piecewise-geometric shrinking tolerance band ---
    # Log-space interpolation => every segment is a geometric climb (constant
    # band-contraction rate): near-flat ADMM penalty through the hot hold,
    # delayed ramp onto the ALM scale, gentle polish rise, terminal
    # restoration ending at the proven 5*alpha0*D/gamma_min.
    a_x = jnp.asarray(_A_X)
    a_y = jnp.stack([c * one for c in _A_Y] + [_TERM_GAIN * Dj / gmin])
    alpha = alpha0 * jnp.exp(jnp.interp(frac, a_x, jnp.log(a_y)))

    # --- betas: proven beta2 transition + one-cycle beta1 hill with gate ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    rise = 1.0 / (1.0 + jnp.exp(-(frac - _B1_RC) / _B1_RW))
    b1_hill = _B1_BASE + (_B1_POLISH - _B1_BASE) * rise
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _B1_GC) / _B1_GW))
    beta1 = b1_hill + (_B1_LO - b1_hill) * gate

    return lr, alpha, beta1, beta2