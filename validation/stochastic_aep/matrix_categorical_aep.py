"""Stochastic AEP estimator for the 48-cell matrix.

Each cell has a discrete wind rose with (direction, speed, weight) triples. We
sample K=50 (sector_index) ~ Categorical(weights), then use that sector's
(direction, speed) as the (wd, ws) draw. The estimator is unbiased of the
deterministic full-sum AEP that the original matrix computed.

For roses where a single sector center has multiple speed entries (uniform: 1
direction × 24 speeds; omnidir: 24 directions × 1 speed each), the same
categorical sampling works because the rose is already represented as 24 (or 12)
(wd, ws, weight) triples summing to 1.

This is the matrix-specific analogue of Part 2's Weibull-sampling estimator —
the substantive change is sampling from the discrete rose rather than the
continuous Weibull, but both are K-sample unbiased estimators of the same
underlying AEP integral.
"""
import jax
import jax.numpy as jnp


def categorical_rose_aep_factory(sim, wind_rose):
    """Return aep_fn(x, y, key, K) — unbiased categorical K-sample AEP in GWh.

    wind_rose: dict with 'directions_deg', 'speeds_ms', 'weights' (length S
    each; weights sum to 1). All length-S arrays come straight from the
    cell's problem JSON.
    """
    dirs = jnp.array(wind_rose['directions_deg'], dtype=jnp.float64)
    speeds = jnp.array(wind_rose['speeds_ms'], dtype=jnp.float64)
    weights = jnp.array(wind_rose['weights'], dtype=jnp.float64)
    weights = weights / jnp.sum(weights)  # safety: normalize
    S = int(weights.shape[0])

    def aep_fn(x, y, key, K):
        idx = jax.random.choice(key, S, (K,), p=weights)
        wd_samp = dirs[idx]
        ws_samp = speeds[idx]
        ti_samp = jnp.full_like(ws_samp, 0.06)
        r = sim(x, y, ws_amb=ws_samp, wd_amb=wd_samp, ti_amb=ti_samp)
        probs = jnp.full((K,), 1.0 / K)
        return r.aep(probabilities=probs)

    return aep_fn


def deterministic_full_rose_aep(sim, x, y, wind_rose):
    """Deterministic AEP over the full discrete rose (matches the original
    matrix's eval).  Sum_s weight_s * P_total(wd_s, ws_s) * 8760."""
    dirs = jnp.array(wind_rose['directions_deg'], dtype=jnp.float64)
    speeds = jnp.array(wind_rose['speeds_ms'], dtype=jnp.float64)
    weights = jnp.array(wind_rose['weights'], dtype=jnp.float64)
    weights = weights / jnp.sum(weights)
    ti = jnp.full_like(speeds, 0.06)
    r = sim(x, y, ws_amb=speeds, wd_amb=dirs, ti_amb=ti)
    return float(r.aep(probabilities=weights))
