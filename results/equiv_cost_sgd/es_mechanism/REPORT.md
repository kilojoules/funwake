# Why Quick-2023 early stopping "beat" the full 6000-iteration run

**Verdict up front.** It didn't. Extended to 10 seeds, the ES−full AEP delta
is a zero-mean lottery: mean −0.007 GWh, std 0.450, range [−0.637, +0.651],
exactly 5 seeds positive / 5 negative. The +0.18 / +0.65 GWh observation on
seeds 0 and 1 was a two-seed coincidence (P = 1/4 under the symmetric
distribution actually observed). The mechanistic question that remains — and
that this study answers — is why *any* delta of ±0.5 GWh arises between two
runs whose trajectories are bit-identical for 4079 of 6000 steps and whose
endpoints are both feasible to centimeter precision. The answer is an
**endpoint lottery on a noisy plateau**, with two quantified noise sources of
similar size: a genuine ±0.36 GWh micro-basin scatter produced by the tail's
random walk, and a previously unknown ±0.29 GWh **discrete evaluation
texture** in the AEP objective itself (wake-cone masking), which contaminates
every single-point AEP comparison on this stack.

All trajectory claims below are backed by a scan replay of the solver's own
`_sgd_step` (solver untouched) that is **bit-identical** to capped
`topfarm_sgd_solve` runs: max |position difference| = 0.0 m at both step 4079
(ES activation) and step 6000, on every seed checked (0, 1, 5).

---

## 1. Delta distribution over 10 seeds (`paired_10seeds.json`)

| seed | AEP full | AEP ES | delta (ES−full) | pen full (m²) | pen ES |
|------|----------|--------|-----------------|---------------|--------|
| 0 | 5524.716 | 5524.896 | **+0.179** | 2.1e-4 | 0 |
| 1 | 5518.424 | 5519.074 | **+0.651** | 2.1e-4 | 0 |
| 2 | 5522.638 | 5522.071 | **−0.567** | 1.2e-4 | 0 |
| 3 | 5520.846 | 5521.408 | **+0.562** | 1.6e-4 | 0 |
| 4 | 5519.103 | 5519.550 | **+0.447** | 8.0e-5 | 0 |
| 5 | 5527.299 | 5526.662 | **−0.637** | 1.9e-4 | 0 |
| 6 | 5528.674 | 5528.339 | **−0.335** | 2.6e-4 | 0 |
| 7 | 5525.457 | 5525.036 | **−0.420** | 1.4e-4 | 0 |
| 8 | 5534.446 | 5534.330 | **−0.115** | 1.7e-4 | 0 |
| 9 | 5521.720 | 5521.885 | **+0.165** | 3.1e-4 | 0 |

Mean −0.007, std 0.450, median +0.025, 5+/5−. For scale: the cross-seed
(basin-to-basin) AEP spread is ±4.6 GWh, ~10× larger.

## 2. Shape of AEP(t) over the tail (`tail_curves.json`, seeds 0, 1, 5)

The tail is **not** a decline, an oscillating decay, or a drop-then-flat. It
is a **stationary noise band**: AEP(t) fluctuates with std 0.23–0.48 GWh
(peak-to-peak 1.2–2.6 GWh) and a *flat mean* from before activation to the
end (seed 0: window means 5524.64 → 5524.70 → 5524.76 → 5524.72 → 5524.71 →
5524.70 across lr ≈ 5.5 → 0.016). No AEP is systematically "lost" anywhere in
the tail; the endpoints are unremarkable draws from the band — the full
endpoint lands at the 49th / 27th / 93rd percentile of its own tail band
(seeds 0/1/5), the ES endpoint at the 75th / 96th / 27th. Tail net drift
(end − activation): +0.072, −0.429, +0.425 GWh — sign varies by seed.

Counterfactual tails from the exact step-4079 state discriminate the forces:

- **frozen** (lr = 5, alpha frozen; 1921 more steps): stationary band, no
  degradation (seed 0 mean stays 5524.6, band ±0.5). Kills mechanism (a) —
  large near-sign steps do *not* progressively walk the layout into a worse
  basin.
- **aep_only** (constraint gradient zeroed): AEP climbs +15.7 GWh as boundary
  turbines escape the polygon (final penalty 1.1e7 m²). The boundary binds
  hard; boundary turbines sit in a strong outward-AEP vs inward-penalty force
  balance.
- **es_like** (AEP gradient zeroed, i.e., ES without its break): converges in
  ~10 steps via momentum flush, penalty → exactly 0, AEP change +0.12
  (texture-scale). Confirms ES's cleanup is a tiny, essentially neutral
  perturbation.

**Boundary limit cycle (refines an "established fact").** At activation the
layout is NOT centimeter-feasible: penalty is 5–81 m² (mean 28), with 1–6
turbines riding meters *outside* the fence (seed 0: min signed distance −3 to
−7 m). This is the lr-scale equilibrium of the fence force balance; the
excursion amplitude decays ∝ lr, reaching 2e-4 m² (≈1.4 cm) only in the last
~700 steps. So the full run's tail is largely a *slow annealed feasibility
restoration* of a few boundary turbines, while ES restores exact feasibility
instantly with 2–4 constraint-only steps. Endpoint feasibility differs by
centimeters either way — irrelevant to AEP, as previously established.

## 3. Displacement and attribution (`displacement.json`)

- ES vs full endpoints differ **everywhere**: 49.9/50 turbines displaced
  >1 m per seed (median 14–19 m, max 44–78 m; ~2 turbines >100 m).
- ES's own contribution is negligible: the ES endpoint sits median 1.5–1.6 m
  (max 8.4 m) from the activation point; the full endpoint sits median
  14–19 m from it. The delta is created by the full run's extra 1921 steps,
  not by ES's cleanup.
- Displacement correlates **positively** with distance from the boundary
  (Spearman +0.33 mean over 10 seeds; +0.15 to +0.56 in 9/10): the
  **interior** turbines wander most (flat wake plateau), the boundary
  turbines are pinned by the fence in both runs. Kills mechanism (b) —
  boundary-adjacent turbines are not the delta carriers.
- Hybrid transplants (full layout with top-k most-displaced turbines moved to
  ES positions; seeds 0, 1, 5) are **diffuse and non-additive**: k=1 can
  overshoot the total delta 2× or carry the wrong sign (seed 0: k=1 →
  −0.25 GWh, k=2 → +0.37 vs total +0.18; seed 1: k=10 → −0.41 vs total
  +0.65). No small subset of turbines explains the delta; it emerges from
  wake-coupled cm–10 m differences across the whole farm.

## 4. The two noise sources, quantified

### 4a. Evaluation texture (new finding, `texture_probe.json`)

The raw AEP objective is **locally piecewise-constant with discrete jumps**:

- Single-turbine line scans (±5 cm, 2 mm resolution, 5 turbines): smooth
  variation < 1.2e-4 GWh, but 1–3 discrete steps of **0.015–0.118 GWh** each.
- Random perturbation response is non-scaling: std 0.22 / 0.34 / 0.32 GWh for
  ±1 cm / 10 cm / 1 m coordinate perturbations — a discontinuity web, not a
  gradient response.
- Not solver truncation: identical to 1.5e-8 GWh at fpi_tol 1e-6 vs 1e-12.
- Source: `pixwake/deficit/base.py` applies a hard wake-cone mask
  `(dw > 0) & (|cw| < wake_radius = 2σ)`. The Gaussian deficit at the cone
  edge is e⁻² ≈ 13.5% of centerline — a finite deficit switched on/off when
  any (source, receiver) pair crosses a cone edge in any of the 24
  directions.

Texture std at an endpoint ≈ **0.29 GWh** (measured over 128 ±0.25 m
perturbations, all 10 seeds ×3 layouts). It also explains why the dense
AEP(t) band does not narrow as lr → 0.01: consecutive cm-scale steps keep
crossing mask edges (measured per-step |ΔAEP| ≈ 0.4 GWh even at 1 cm steps).

**Consequence beyond this study**: any single-evaluation AEP comparison on
this stack with margins ≲ 0.5 GWh (≈1e-4 relative) is dominated by texture,
not layout quality.

### 4b. Real micro-basin scatter (`smoothed_deltas.json`)

Averaging AEP over 128 random ±0.25 m perturbations (SE ≈ 0.007 GWh) removes
the texture and measures the smooth landscape:

- **Smoothed delta (ES − full): mean −0.100, std 0.356, 4/10 positive**
  (mean SE ±0.113 → consistent with zero; if anything the full run is
  slightly better). corr(raw, smoothed) = 0.854; e.g. seed 3's raw +0.562
  was almost entirely texture (smoothed −0.048).
- Decomposition at the activation point:
  smoothed cleanup (ES − act) = **−0.003 ± 0.161** (5/10 positive) — ES's
  momentum-flush/pullback is exactly neutral;
  smoothed tail drift (full − act) = **+0.097 ± 0.252** (6/10 positive) —
  the annealed tail is a slight *improvement* on average, not a loss.

## 5. Mechanism (synthesis)

By step 4079 the optimization is converged in distribution: the layout rides
a stationary Adam limit cycle (betas 0.1/0.2 make every step ≈ lr per
turbine) on a nearly flat constrained plateau, with a few boundary turbines
oscillating meters outside the fence in an lr-scale force balance. The
remaining 1921 steps change no systematic quantity: they anneal the cycle
amplitude to zero while the interior turbines random-walk a further ~15 m and
the boundary turbines converge onto the fence. **ES and the full run
therefore sample two different endpoints of the same stationary
distribution** ~1900 steps apart; their AEP difference is a zero-mean draw
composed of ±0.36 GWh of genuine micro-basin scatter plus ±0.25 GWh of
evaluation texture (raw delta std 0.45 ≈ √(0.36² + 0.25²)). Candidate
mechanisms (a) "lr≈5 steps walk into a worse basin, annealing locks it in"
and (b) "boundary oscillation settles suboptimally" are both refuted by
direct measurement (frozen-schedule stationarity + slightly positive smoothed
drift for (a); positive displacement-vs-boundary-distance correlation,
fence-pinned boundary turbines in both runs, and diffuse non-additive hybrid
attribution for (b)).

Practical reading: ES buys the ~32% iteration saving at zero expected AEP
cost with *better* endpoint feasibility (penalty exactly 0 vs 2e-4 m²), but
it confers no AEP advantage; per-seed deltas of either sign up to ~0.65 GWh
are inherent endpoint noise, half of which is not even physical.

## Files

Scripts (repo `tools/`): `es_mechanism_paired.py` (exp 1),
`es_mechanism_tail.py` (exp 2 + counterfactual variants + bit-identity
verification), `es_mechanism_displace.py` (exp 3),
`es_mechanism_actpoint.py` (activation decomposition),
`es_mechanism_bandstats.py` (band statistics / freeze check),
`es_mechanism_smoothed.py` (exp 4), `es_mechanism_texture.py` (exp 5).

Data (this directory): `paired_10seeds.json`, `tail_curves.json` (seeds
0, 1, 5; dense per-step tail curves + variants), `displacement.json`,
`activation_decomposition.json`, `band_stats.json`, `smoothed_deltas.json`,
`texture_probe.json`.
