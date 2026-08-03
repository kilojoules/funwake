import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-hold best (+0.0577%): the MOMENT DYNAMICS
# gain phases. The lineage explored the lr and alpha waveforms exhaustively
# (cosine restarts -> flat-top hold -> overdrive spikes, which LOST ground at
# +0.0461%), but beta1 has never left [0.02, 0.1] and beta2 has only ever made
# its single 0.2 -> 0.9 cool-down step. This generation keeps the winning lr
# and alpha machinery BIT-FOR-BIT (the parent's losing overdrive spikes are
# dropped, reverting to the best's clean flat-top waveform) and instead
# PHASE-TRANSITIONS THE ADAM MOMENTS — the prior-art menu's bet #4 fused with
# the one CLAUDE.md ablation never run (standard Adam 0.9/0.999 vs TopFarm
# 0.1/0.2), applied per-phase rather than globally:
#
#   BALLISTIC EXPLORATION — on the hot flat-top hold, beta1=0.85, beta2=0.95
#   (standard-Adam regime; beta1 < sqrt(beta2) stability condition holds
#   everywhere along the blend). Adam's normalized update means high momentum
#   does NOT amplify the metre-scale step size — it ALIGNS consecutive steps,
#   so turbines travel in coherent ~1/(1-beta1) ~ 7-step straight runs instead
#   of dithering in place. Net displacement per unit heat rises severalfold:
#   more basin-hopping from the SAME proven 1.5*D -> 1.1*D heat budget, which
#   is exactly the lever left after the hotter-spike route failed.
#
#   FAST-TWITCH REPAIR — inside each cold notch both moments collapse to the
#   native TopFarm regime (beta1 -> 0.05, beta2 -> 0.2): momentum re-anchors
#   to the current gradient in a single step, so no exploration momentum drags
#   turbines back across the boundary mid-repair, and v tracks the alpha
#   burst's constraint gradients instantly — the proven repair dynamics,
#   now with a far larger hold/notch contrast.
#
#   PROVEN POLISH + ENDGAME — from the 62% cool-down on, the moments revert to
#   the lineage's proven (0.1, 0.9) and the entire 5/5-seed-feasible endgame
#   runs unchanged: straight linear lr tail landing exactly on gamma_min,
#   logistic ramp to the bounded 6*alpha0 ALM plateau, cubic-delayed geometric
#   climb from 78% to the terminal 5*alpha0*D/gamma_min feasibility spike,
#   with the gated beta1 -> 0.02 drop while it fires.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three repair notches inside the exploration phase
_Q = 6.0              # notch = sin^(2Q); ~80% of each cycle at full hold lr
_HI0 = 1.5            # initial hold level, in units of D (proven)
_HI1 = 1.1            # final hold level; the linear tail starts from here (proven)
_LO = 0.35            # notch-bottom lr — deep, surgical repair windows (proven)
_A_LO = 0.4           # exploration penalty floor, in alpha0 units (proven)
_A_B0 = 3.0           # first restoration burst height, in alpha0 units (proven)
_A_B1 = 8.0           # last restoration burst height, in alpha0 units (proven)
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_HOLD = 0.95       # NEW: standard-Adam second moment on the hot hold
_B2_LO = 0.2          # native fast-twitch v inside repair notches (proven value)
_B2_HI = 0.9          # proven polish/endgame value from cool-down on
_B2_CENTER = 0.62     # moment hand-over aligned with the cool-down start (proven)
_B2_WIDTH = 0.05
_B1_HOLD = 0.85       # NEW: ballistic exploration momentum on the hold
_B1_NOTCH = 0.05      # proven repair momentum inside each notch
_B1_POLISH = 0.1      # proven native momentum from cool-down through polish
_B1_LO = 0.02         # proven near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> flat-top hold with three deep narrow notches -> tail ---
    # Bit-for-bit the +0.0577% best. notch = sin(pi*N*fc)^(2Q): 0 at fc=0 and
    # fc=1 (the tail launches from the clean hold level _HI1*D), ~0 for ~80%
    # of each cycle, 1 briefly at each cycle midpoint; fc freezes past _F_COOL.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    notch = jnp.sin(jnp.pi * _N_CYC * fc) ** (2.0 * _Q)
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying hold envelope
    lr_hold = (_LO + (hi - _LO) * (1.0 - notch)) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + notch-synchronized growing bursts -> plateau -> climb ---
    # Bit-for-bit the proven feasibility machinery: bursts fire exactly inside
    # the lr notches (repair when steps are small), vanish for frac >= _F_COOL,
    # then the bounded ALM plateau and cubic-delayed terminal spike take over.
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * notch
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: NEW phase-transitioned moments ---
    # Hold: ballistic standard-Adam (0.85, 0.95). Notch: native fast-twitch
    # repair (0.05, 0.2) — both moments ride the same smooth notch waveform, so
    # the hand-offs are traceable and branch-free. The proven cool-down
    # logistic then blends both moments to the proven polish values (0.1, 0.9),
    # and the proven gated drop takes beta1 to 0.02 for the terminal spike.
    cool = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    b2_exp = _B2_HOLD - (_B2_HOLD - _B2_LO) * notch           # 0.95 on hold, 0.2 in notch
    beta2 = b2_exp + (_B2_HI - b2_exp) * cool                 # -> 0.9 after cool-down
    b1_exp = _B1_HOLD - (_B1_HOLD - _B1_NOTCH) * notch        # 0.85 on hold, 0.05 in notch
    b1_mid = b1_exp + (_B1_POLISH - b1_exp) * cool            # -> 0.1 after cool-down
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * b1r                  # gated terminal drop

    return lr, alpha, beta1, beta2