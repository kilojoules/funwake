import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-notch best (+0.0577%): a DEMON MOMENTUM
# RAMP on beta1 — the one prior-art menu direction (§4: increasing-Sutskever
# vs decaying-Demon momentum; §2: schedule the moments WITH the phases) that
# every schedule in this lineage has left untouched. All previous designs run
# exploration at the native beta1 = 0.1, i.e. near-memoryless steps: at a hot
# lr each turbine jitters against ragged wake-interference gradients instead
# of drifting coherently across basins. Here beta1 itself becomes a phase
# variable:
#
#   beta1 — DEMON DECAY 0.75 -> 0.1 linearly across the exploration phase.
#           Early on, ~4-step gradient averaging turns the hot hold into
#           coherent long-range transport (momentum as a low-pass filter on
#           the wake landscape), which is a genuinely different exploration
#           mechanism, not a hotter version of the old one. The ramp lands
#           exactly on the proven 0.1 at the cool-down start, so the ENTIRE
#           proven polish/feasibility endgame runs on untouched momentum.
#           Inside each repair notch beta1 dips to 0.05 (momentum must never
#           carry turbines back across the boundary mid-repair — this also
#           periodically flushes the accumulated hot-phase momentum), and the
#           proven terminal gate to 0.02 guards the alpha spike.
#   lr    — the proven winning waveform, pushed hotter per the parent
#           guidance ("higher/longer lr peak early"): 3% warmup -> flat-top
#           hold with envelope decaying 1.7*D -> 1.1*D (vs 1.5*D; momentum
#           smoothing is what makes the extra heat usable) across three
#           narrow sin^12 notches down to 0.35*D -> exploration ends at the
#           proven 62% -> proven straight linear tail landing exactly on
#           gamma_min at the last step.
#   alpha — proven machinery intact, floor nudged greedier: 0.35*alpha0
#           exploration floor, growing restoration bursts (3 -> 8 alpha0)
#           synchronized with the lr notches, logistic ramp to the bounded
#           6*alpha0 ALM plateau at 66%, and the 5/5-seed-feasible
#           cubic-delayed geometric climb from 78% to the terminal
#           5*alpha0*D/gamma_min spike.
#   beta2 — proven transition only: 0.2 -> 0.9 logistic at the cool-down.
#
# Clean ablation: only the exploration mechanism (momentum + heat) changes;
# repair cadence, penalty endgame, and the polish phase are bit-for-bit the
# proven feasible design.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three repair notches inside the exploration phase
_Q = 6.0              # notch = sin^(2Q); ~80% of each cycle at full hold lr
_HI0 = 1.7            # initial hold level, in units of D — hotter than any tried
_HI1 = 1.1            # final hold level; the linear tail starts from here
_LO = 0.35            # notch-bottom lr — deep, surgical repair windows (proven)
_A_LO = 0.35          # exploration penalty floor, in alpha0 units (greedier)
_A_B0 = 3.0           # first restoration burst height, in alpha0 units (proven)
_A_B1 = 8.0           # last restoration burst height, in alpha0 units (proven)
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
_B1_START = 0.75      # Demon momentum at step 0 — coherent basin transport
_B1_POLISH = 0.1      # Demon ramp lands on the proven native momentum here
_B1_NOTCH = 0.05      # momentum flushed inside each repair notch (proven dip)
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> hotter flat-top hold with three deep narrow notches -> tail ---
    # notch = sin(pi*N*fc)^(2Q): 0 at fc=0 and fc=1 (the tail launches from the
    # clean hold level _HI1*D), ~0 for ~80% of each cycle, 1 briefly at each
    # cycle midpoint. fc freezes at 1 past _F_COOL.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    notch = jnp.sin(jnp.pi * _N_CYC * fc) ** (2.0 * _Q)
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying hold envelope
    lr_hold = (_LO + (hi - _LO) * (1.0 - notch)) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + notch-synchronized growing bursts -> plateau -> climb ---
    # Bursts fire exactly inside the lr notches (repair when steps are small)
    # and vanish for frac >= _F_COOL; the proven bounded endgame then takes over.
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * notch
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: Demon beta1 ramp across exploration + proven transitions ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_demon = _B1_START + (_B1_POLISH - _B1_START) * fc      # 0.75 -> 0.1, frozen after cool-down
    b1_exp = b1_demon - (b1_demon - _B1_NOTCH) * notch        # flushed while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2