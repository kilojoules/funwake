# FunWake-2 — Prior-Art Survey (FROZEN)

**Status: FROZEN — 2026-08-01.** Do not edit in place; supersede with a dated
successor if it needs revision. Compiled from a four-track web-verified literature
review (learning-rate schedules; Adam-moment scheduling & LR-free/learned optimizers;
constraint-penalty scheduling; wind-farm layout optimization + the TopFarm-SGD
baseline; LLM-driven program discovery & quality-diversity).

**Purpose.** (1) A design-rationale reference for the schedule-discovery framework and
the paper's related work; (2) an *optional, firewall-safe static context* that can be
injected into the mutator's scoped workspace to ground candidate `schedule_fn`s in
named prior art. **Firewall note:** this document contains ONLY public literature, the
schedule interface, and the *published* native recipe (Quick et al. 2023). It contains
NO held-out/test AEP values, no test-cell identities, and no test-decision or pre-registration content
— so it is safe to place in the mutator scope. Keep it that way in any successor.

---

## 0. The design problem this literature serves

The searched object is
```python
schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0)
    -> (lr, alpha, beta1, beta2)
```
driving a TopFarm-style Adam optimizer of turbine (x,y) that maximizes AEP subject to
a signed-distance polygon/inclusion-zone boundary constraint and a minimum-spacing
constraint (combined gradient `grad_obj + alpha·grad_constraint`). `lr` is built from
the rotor diameter `D` and decays toward `gamma_min` (a metre-valued feasibility
tolerance = terminal lr); `alpha` is the penalty weight; `beta1,beta2` are the Adam
moment-decay coefficients. `alpha0 = mean|∇AEP|/D` is supplied.

**The published native baseline the search generalizes** (Quick et al. 2023, below):
product/harmonic lr decay `η(t)=η₀·Π 1/(1+iδ)` from `η₀≈D/5` (v1) / `c·D` (v2) to a
small terminal lr; penalty *coupled inversely to the step size* `α(t)=α₀·η₀/η(t)` (so
penalty rises as lr decays); short-memory Adam `(β₁,β₂)=(0.1,0.2)`. The open research
question is whether **decoupled or non-monotonic** schedules of these four knobs beat
this hard-coded coupling.

---

## 1. Learning-rate decay families

- **Inverse-time / `1/(1+kt)` (Robbins & Monro 1951, *Ann. Math. Stat.* 22:400–407).**
  Stochastic-approximation convergence needs `Ση=∞, Ση²<∞` — the ancestor of every
  decay below. *lr:* the cleanest peak→terminal law
  `lr = gamma_min + (c·D − gamma_min)/(1+k·step)`; the native product-decay is this
  family. *(Canonical, non-arXiv.)*
- **Step decay (geometric) (Ge, Kakade, Kidambi, Netrapalli 2019, NeurIPS, arXiv:1904.12838).**
  Geometrically-spaced drops are near-minimax-optimal for the *last iterate* of SGD on
  least squares; polynomial decay of the last iterate is provably sub-optimal. *lr:*
  `lr = c·D · factor**floor(step/period)` clamped at `gamma_min`; a caution that smooth
  poly-decay may under-perform discrete drops for final-layout quality.
- **Exponential decay.** `lr=c·D·exp(−λ·step)` with `λ=ln(c·D/gamma_min)/total_steps`
  to hit `gamma_min` exactly. *(Standard; no single origin paper.)*
- **Polynomial ("poly") decay.** `lr=(c·D−gamma_min)(1−step/total_steps)**p+gamma_min`;
  `p=1` linear, `p=2` quadratic (DeepLab/BERT usage). *lr:* natural bounded schedule
  given a hard terminal `gamma_min`; `p` tunes how long you linger at the `c·D` peak.

## 2. Cosine annealing, warm restarts, cyclical & one-cycle

- **SGDR: cosine annealing + warm restarts (Loshchilov & Hutter 2017, ICLR, arXiv:1608.03983).**
  `lr = gamma_min + ½(c·D−gamma_min)(1+cos(π·t_cur/T_i))`, restart to max each cycle
  (`T_mult`). *lr (and by analogy alpha):* restarts re-explore the non-convex AEP
  surface / escape locally-feasible layouts. **Embodied in the `cyclic`/`cosine` seeds.**
- **Cyclical Learning Rates (Smith 2017, WACV, arXiv:1506.01186).** Triangular
  oscillation between base/max lr + the "LR range test." *lr:* periodic large-step
  exploration without a hand-set decay; range test → calibrate `max_lr` from `D`.
- **One-cycle / super-convergence (Smith & Topin 2018, arXiv:1708.07120).** *One*
  warmup→large-peak→decay cycle with **momentum cycled inversely** (high momentum when
  lr low). *lr + beta1:* the single most transplantable lr+β1 recipe — peak at `c·D`,
  end at `gamma_min`, and move `beta1` anti-correlated to lr. **[shortlist]**

## 3. Warmup (linear / inverse-√ / theory)

- **Linear warmup + linear scaling (Goyal et al. 2017, arXiv:1706.02677).** Ramp lr
  `0→peak` over the first W steps to survive the unstable early phase. *lr:* a short
  `lr=(step/W)·c·D` prevents blowing turbines outside the polygon on step 1 (empty Adam
  moments).
- **Inverse-√ "Noam" (Vaswani et al. 2017, NeurIPS, arXiv:1706.03762).**
  `lr∝min(t^{-0.5}, t·W^{-1.5})`: closed-form warmup+decay, no branching, differentiable.
- **Why warmup? (Kalra & Barkeshli 2024, NeurIPS, arXiv:2406.09405).** Warmup drives a
  sharpness-reduction phase that *enables* the later large lr. *lr:* motivates genuine
  warmup even here — the AEP+penalty landscape is sharp at the initial grid layout.
- **A Closer Look at DL Heuristics (Gotmare et al. 2019, ICLR, arXiv:1810.13243).**
  Warmup mainly prevents deep-layer instability; several cosine-restart folk-explanations
  don't hold. *Meta:* treat schedule folklore as hypotheses to A/B test, not truth.

## 4. Adam and β1/β2 scheduling

- **Adam (Kingma & Ba 2015, ICLR, arXiv:1412.6980).** Bias-corrected 1st(β1)/2nd(β2)
  moment EMAs; defaults 0.9/0.999. The object being scheduled; bias correction already
  does an implicit early warmup.
- **AdaGrad (Duchi, Hazan, Singer 2011, JMLR 12:2121).** β2→1 limit; monotone shrink.
  *beta2:* the β2→1 endpoint = maximal memory / smallest adaptive lr late = "freeze the
  layout" terminal regime.
- **RMSprop (Tieleman & Hinton 2012, Coursera 6.5).** The β2 EMA Adam inherited.
  *(Unpublished lecture.)*
- **AMSGrad (Reddi, Kale, Kumar 2018, ICLR best paper).** Running *max* of 2nd moment →
  non-increasing effective lr. *beta2:* the principled fix for late-stage constraint
  oscillation (or a β2→1 ramp).
- **AdamW (Loshchilov & Hutter 2019, ICLR, arXiv:1711.05101).** Decouple weight decay
  from the adaptive step. *Relevance:* orthogonal (no weight decay here), but the
  "decouple magnitudes that shouldn't interact" lesson supports decoupling `alpha` from
  `lr` rescaling.
- **RAdam (Liu et al. 2020, ICLR, arXiv:1908.03265).** The adaptive lr has pathological
  early variance (few 2nd-moment samples); warmup/rectification fixes it. *beta2 + lr:*
  the strongest case for **ramping β2** (start low ~0.2–0.9 → 0.999) or warming up lr —
  directly justifies "betas are ramped." **[shortlist]**
- **Nadam (Dozat 2016, ICLR workshop).** Nesterov look-ahead on Adam's β1 term.
  *beta1:* the β1 pathway is where look-ahead pays off.
- **Increasing-momentum schedule (Sutskever et al. 2013, ICML).** Slowly increase
  momentum 0.5→0.99. *beta1:* schedule β1 *up* — low while exploring layouts, high late
  to average noise. Opposite end from TopFarm's 0.1.
- **Demon: decaying momentum (Chen et al. 2019, arXiv:1910.04952).** Closed-form β1
  *decay* over training. *beta1:* ready-made monotone β1 form — note it *decays* where
  Sutskever *increases*, so both directions bridge TopFarm's 0.1 and Adam's 0.9 and are
  cheap to A/B test. **[shortlist: β1 both directions]**
- **YellowFin (Zhang & Mitliagkas 2019, MLSys, arXiv:1706.03471).** Auto-tunes one
  momentum+lr from local curvature/variance. *Relevance:* aspirational — FunWake's
  signature exposes only `step`, so closed-loop tuning isn't directly implementable.

## 5. LR-free / adaptive-scale & learned optimizers

- **D-Adaptation (Defazio & Mishchenko 2023, ICML outstanding, arXiv:2301.07733).**
  Online lower-bound the distance-to-solution `D` and set the step from it — no lr to
  tune. *lr:* the whole schedule could be *derived* from a scale estimate; note the
  coincidence that their `D` and FunWake's `D` (rotor diameter) both set the step scale.
- **Prodigy (Mishchenko & Defazio 2023, arXiv:2306.06101).** Faster D-estimate;
  often beats tuned Adam. *lr:* the SOTA "no-lr" template if `schedule_fn` ever adapts.
- **DoG (Ivgi, Hinder, Carmon 2023, ICML, arXiv:2302.12022).** step = (max distance
  travelled)/(√Σ‖g‖²) — no lr. *lr:* the distance/grad-norm ratio is a self-scaling lr;
  "distance travelled" is metres, aligning units with `gamma_min`.
- **Schedule-Free (Defazio et al. 2024, NeurIPS, arXiv:2405.15682).** Polyak–Ruppert
  averaging replaces the decay schedule; no `total_steps` needed; still likes a short
  warmup. *lr/beta1:* suggests reporting an *averaged* (feasible) layout, not the last
  iterate; note reported (β1,β2) sensitivity — betas still matter.
- **Hypergradient descent (Baydin et al. 2018, ICLR, arXiv:1703.04782).** Update lr by
  gradient descent on lr via `⟨g_t,g_{t−1}⟩`. *lr:* grow lr when consecutive gradients
  align, shrink when they conflict.
- **Learning to learn by GD by GD (Andrychowicz et al. 2016, NeurIPS).** Meta-learned
  LSTM optimizer. *Meta:* the intellectual parent of FunWake's premise (an outer loop
  discovering optimizer behavior).
- **L2O: A Primer and Benchmark (Chen et al. 2022, JMLR, arXiv:2103.12828).** Survey +
  Open-L2O; documents L2O's poor OOD generalization. *Meta:* a direct warning for
  train-on-one-farm / test-on-another — prefer robust parametric families (§7) over
  highly-tuned exotic curves.
- **VeLO (Metz et al. 2022, arXiv:2211.09760).** HP-free learned optimizer at
  ~4000 TPU-months. *Meta:* fully-learned schedules are feasible but expensive —
  supports the cheaper LLM-in-the-loop alternative.

## 6. Modern schedule-shape results (peak→terminal)

- **Optimal linear-decay LR (Defazio, Cutkosky, Mehta, Mishchenko 2023, arXiv:2310.07831).**
  Theory: optimal step ∝ `(1−step/total_steps)`; beats cosine on many workloads.
  *lr:* `lr = gamma_min + (c·D−gamma_min)(1−step/total_steps)`. **[shortlist]**
- **Straight to Zero / D2Z (Bergsma et al. 2025, ICLR, arXiv:2502.15938).** Linear
  decay-to-zero beats cosine across scales; explained via AdamW-as-EMA early-vs-late
  balance. *lr:* corroborates linear decay; clean mental model for how fast to leave the
  peak.
- **Warmup-Stable-Decay / WSD (Hu et al. 2024 MiniCPM arXiv:2404.06395; Hägele et al.
  2024, NeurIPS spotlight, arXiv:2405.18392).** warmup → long *constant* lr → short
  sharp cooldown; matches cosine with an arbitrary stop point; loss drops mostly during
  cooldown. *lr + alpha:* **strong template** — hold lr near `c·D` most of the budget,
  cool to `gamma_min` in the final ~20%; the late cooldown is exactly when to crank
  `alpha` to snap into feasibility. **[shortlist]**

---

## 7. Constraint handling & penalty (α) scheduling — the richest section for FunWake

**Framing.** `schedule_fn` is *open-loop*: it sees the step counter and static problem
constants, NOT the current layout / AEP / violation. Almost all methods below are
*closed-loop*. The lesson is rarely "copy the update rule" — it is **shape `alpha(t)`
(and its coupling to `lr(t)` and `gamma_min`) to reproduce, on the typical trajectory,
what the closed-loop method would have done.** The native `α(t)=α₀·D/lr(t)` is one such
open-loop translation (penalty continuation ramped inversely with the step).

### 7.1 Exterior quadratic penalty & continuation
- **Quadratic (Courant) penalty (Courant 1943; Fiacco & McCormick 1968; Nocedal &
  Wright Ch.17).** `f + (μ/2)Σc²`, `μ→∞`, and `μ·c → λ*` (the multipliers). *alpha:*
  this **is** the native scheme (`alpha≡μ`). The multiplier fact tells you the *right
  terminal magnitude*: `alpha·(constraint grad)` should match the AEP-grad scale at the
  active boundary — exactly what `alpha0` calibrates.
- **Ill-conditioning as `μ→∞` (Nocedal & Wright Ch.17; Bertsekas 1982).** The penalized
  Hessian condition number grows like `μ`; near the end the problem is stiff. *alpha:*
  the key warning for aggressive ramps — the combined operator conditioning `~alpha`, so
  at large `alpha` you must shrink `lr` (**this is why `alpha∝1/lr` is well-motivated**)
  or lean on `beta2` to rescale the high-curvature constraint direction. Argues for
  `alpha` that ramps then **plateaus** at a finite level, not `→∞`.

### 7.2 Exact penalties (finite penalty suffices)
- **ℓ1/nonsmooth exact penalty (Zangwill 1967; Pietrzykowski 1969; Han & Mangasarian
  1979; Nocedal & Wright Ch.17).** `f + ν·Σ|c|` is *exact* for any finite `ν > ‖λ*‖∞` —
  **no `ν→∞` needed.** *alpha:* the key non-obvious counter-message to continuation — a
  *finite plateau* `α_max` (a few×`alpha0`) can be feasible AND better-conditioned than
  native's diverging `D/lr`. **[shortlist]**
- **Steering exact penalty (Byrd, Nocedal, Waltz 2008/2012).** Raise `ν` only when
  feasibility stalls. *alpha:* motivates a **late, sharp** ramp, not a smooth early one.

### 7.3 Augmented Lagrangian / Method of Multipliers / ADMM *(deepest, most transferable)*
- **Method of Multipliers / ALM (Hestenes 1969; Powell 1969; Rockafellar 1973–74;
  Bertsekas 1982).** `f + λᵀc + (μ/2)‖c‖²`, then `λ ← λ + μ·c`. **Central result:
  the explicit `λ` term means feasibility is reached at *moderate, bounded* `μ` — no
  divergence, no ill-conditioning.** *alpha:* the most under-volunteered idea. A literal
  ALM is off the table (no multiplier exposed), but the insight reshapes the schedule —
  keep `alpha` **moderate/bounded** and let **momentum act as an implicit multiplier**
  (Adam's β1 accumulates the persistent boundary gradient like `μ·c` accumulates into
  `λ`). Open-loop surrogate: a **two-phase / logistic `alpha(t)`** (low while exploring,
  saturating to a finite plateau). **[shortlist]**
- **Practical ALM penalty-update rules (Conn–Gould–Toint / LANCELOT 1991–92; Andreani et
  al. / ALGENCAN 2007–08).** Increase `μ` *only if* the violation didn't shrink enough;
  else keep `μ`, update `λ`. *alpha:* **don't ramp from step 0** — hold `alpha≈alpha0`
  through exploration, then ramp (geometric `factor≈2` between plateaus mirrors
  LANCELOT). **[shortlist: delayed ramp]**
- **ADMM (Glowinski–Marrocco 1975; Gabay–Mercier 1976; Boyd et al. 2011, FnT ML 3(1):1).**
  Alternating minimization with a *fixed* penalty `ρ`. *alpha:* (1) a **flat `alpha`**
  can be competitive — worth an ablation vs native; (2) the alternating structure
  suggests a **cyclic `alpha`** (feasibility bursts alternating with AEP phases),
  optionally synced to cosine-lr restarts.
- **AL/primal-dual inside deep learning (Fioretto et al. 2020; Park & Van Hentenryck
  2023, AAAI; Sangalli et al. 2023, arXiv:2310.16647).** Closest modern analogue (Adam +
  constraints). Consistent finding: **start small penalty, grow it as training proceeds,
  slow dual does the heavy lifting.** *alpha:* validates a **warm-up on `alpha`** (small
  early); large `alpha` at step 0 destabilizes the objective descent. Bias `beta1` higher
  during the ramp as the dual substitute.

### 7.4 Interior-point / barrier
- **Log-barrier / SUMT (Frisch 1955; Fiacco & McCormick 1968).** Strictly-interior,
  `μ→0` central path. *alpha:* barriers are *interior* (opposite of FunWake's exterior
  penalty), but both are monotone continuation — tuning wisdom transfers. If turbines
  must stay *inside* the polygon at every iterate (wake-model validity), a barrier on the
  signed distance with a *decaying* weight is the principled alternative.
- **IPOPT: barrier + filter line search (Wächter & Biegler 2006, Math. Prog. 106:25).**
  Continuation + filter globalization. *alpha:* template for a hybrid (continuation on
  `alpha` + filter-inspired periodic feasibility emphasis).

### 7.5 Filters, merit functions, feasibility restoration
- **Filter method (Fletcher & Leyffer 2002, Math. Prog. 91:239).** Accept a step if it
  improves objective *or* violation — no penalty parameter; a Pareto `(f,‖c‖)` filter.
  *alpha:* the bi-objective AEP-vs-infeasibility view is FunWake's exact tension —
  motivates low `alpha` early (permit objective-improving, feasibility-worsening
  exploration), rising later.
- **Merit functions (Nocedal & Wright Ch.15,18).** `f + ν·infeasibility`, `ν` must
  exceed the multiplier for descent. *alpha:* a lower bound on useful `alpha` (~`alpha0`)
  and, again, a finite plateau.
- **Feasibility restoration (Fletcher–Leyffer; Wächter–Biegler; funnel/restoration SQP).**
  When too infeasible, *drop the objective* and minimize violation, then resume. *alpha:*
  a concrete novel shape — **periodic restoration bursts** (spike `alpha`, drop `lr`
  briefly), and especially a **terminal feasibility spike** so the returned layout is
  feasible even if it wandered infeasible for better AEP. **[shortlist]**

### 7.6 Projection / proximal / Frank–Wolfe (feasible-set methods)
- **Projected gradient (Rosen 1960; Goldstein 1964; Levitin–Polyak 1966).** Step then
  project onto the feasible set — exact, no penalty. *alpha:* the polygon admits cheap
  projection; spacing does not. Hybrid worth flagging: **project onto the polygon
  exactly** and use `alpha` *only for spacing* — halves what `alpha` balances, letting it
  be smaller/better-conditioned.
- **Proximal gradient (Moreau 1965; Combettes–Wajs 2005; Parikh–Boyd 2014).** Prox =
  projection handles feasibility without a growing penalty. *alpha:* supports a
  bounded/decaying-`alpha` philosophy.
- **Frank–Wolfe (Frank & Wolfe 1956; Jaggi 2013, ICML).** Projection-free; schedule-only
  step `γ_k=2/(k+2)`. *lr/alpha:* its `~1/k` step matches native's product decay; shows
  that with a feasibility-preserving mechanism no penalty ramp is needed.

### 7.7 Homotopy / graduated optimization
- **Graduated Non-Convexity (Blake & Zisserman 1987).** Smooth surrogate → true
  objective. **Gaussian homotopy (Mobahi & Fisher 2015, AAAI).** **Graduated opt for
  stochastic non-convex (Hazan, Levy, Shalev-Shwartz 2016, ICML)** — decreasing smoothing
  `σ`, `O(1/ε²)`. *alpha/lr:* (1) penalty continuation *is* a homotopy (easy→hard) →
  justifies *slow early* `alpha` so the layout finds a good AEP basin first; (2) the
  Monte-Carlo AEP gradient (Quick et al.) *is* a smoothing whose `σ` the `lr`/`beta2`
  schedule should exploit — smooth early, sharpen late.
- **Wake Expansion Continuation / WEC (Thomas & Ning 2018, JPCS; Thomas, McOmber, Ning
  2022, Wind Energy, 10.1002/we.2692).** Start with *widened* wakes (smoothed AEP), shrink
  to the true problem — a domain-specific homotopy that reduces local-optima entrapment.
  *lr/alpha:* strong domain precedent that gradually changing a problem parameter over the
  run helps — the "smooth-then-sharpen" prior for the whole schedule.

### 7.8 Penalty–step-size coupling (the FunWake design axis)
- **Quick et al. 2023 (WES 8:1235) — the native scheme.** `α_i=α₀·(η₀/η_i)`,
  `α₀=mean|∇AEP|/L`, product lr decay `η₀Π1/(1+iδ)`, `η₀=D/5`, `η_T=0.1 m`, betas
  0.1/0.2. *alpha:* well-motivated (conditioning `~alpha` matched by step `~lr`; `1/lr`
  growth → terminal feasibility). **Improvement hypotheses:** (a) *decouple* `alpha` from
  `1/lr` (bounded/logistic; §7.2/7.3); (b) *delay* the ramp (§7.3/7.5); (c) *terminal
  feasibility spike* (§7.5); (d) *co-schedule betas with the alpha phase* (§7.3).
- **Penalty = dual step size (Boyd et al. 2011; primal-dual DL).** In AL-SGD the penalty
  *is* the dual-ascent step. *Relevance:* tying `alpha` and `lr` is principled (primal/
  dual steps of one saddle-point) — the design question is the *form* of the coupling.

### 7.9 Constraint handling in metaheuristics (WFLO is often solved this way)
- **Survey: Coello Coello 2002 (CMAME 191:1245).** Static/dynamic/adaptive/feasibility-
  rule taxonomy.
- **Static / multi-level penalty (Homaifar et al. 1994; guidelines Richardson et al.
  1989).** "Penalty ≈ cost to repair infeasibility, not arbitrarily huge." *alpha:*
  echoes exact-penalty threshold → `alpha` on the scale of "AEP per metre of violation" =
  `alpha0`; against over-large `alpha`.
- **Dynamic (time-dependent) penalty (Joines & Houck 1994, IEEE CEC).**
  `f + (C·t)^α·violation` — penalty grows with the iteration counter. *alpha:* the
  metaheuristic twin of continuation, expressed **purely as a function of the step** =
  FunWake's exact setting → licenses `alpha0·(1+C·t)^p` ramps; sweep `p∈{1,2}`.
- **Adaptive/feedback penalty (Bean & Hadj-Alouane 1992; Lemonge–Barbosa 2004;
  self-adaptive Tessema–Yen 2006).** Raise/lower `alpha` by feasible/infeasible ratio.
  *alpha:* behavioral target for the open-loop schedule — rise through the infeasible
  early/mid phase, *ease* once feasible → **ramp-then-plateau/slight-decay**, and small
  early `alpha` (infeasible exploration helps escape wake local optima).
- **Superiority-of-feasible (Deb 2000, CMAME 186:311); Stochastic Ranking (Runarsson &
  Yao 2000, IEEE TEC 4:284); ε-constrained (Takahama & Sakai 2006, CEC winner).**
  Deb: at the end, feasibility must dominate (final `alpha` decisively large). Stochastic
  ranking: a mildly *stochastic/oscillating* `alpha` that occasionally de-emphasizes
  constraints. **ε-constrained maps directly onto `gamma_min`:** treat feasibility with a
  *generous early tolerance shrinking toward `gamma_min` late*; since `gamma_min` is a
  fixed input, the equivalent lever is an `alpha` that keeps early violations cheap then
  contracts the enforced band to `gamma_min`. **[shortlist]**

---

## 8. Wind-farm layout optimization & benchmark framing

- **Review (Herbert-Acero et al. 2014, *Energies* 7:6930).** Wake/formulation/algorithm
  decomposition; historic reliance on gradient-free metaheuristics; highly multimodal
  landscape → why exploration-vs-feasibility scheduling matters.
- **Boundary-grid reduction (Stanley & Ning 2019, WES 4:663).** 5 params reproduce full
  per-turbine AEP → the landscape is dominated by a few macro-structures → supports
  aggressive early lr to reach a good basin before tightening.
- **Gradient-free lineage:** GA (Mosetti et al. 1994, JWEIA 51:105; Grady et al. 2005,
  Renew. Energy 30:259), binary PSO-TVAC (Pookpunt & Ongsakul 2013, Renew. Energy 55:266),
  random/local search (Feng & Shen 2015, Renew. Energy 78:182). The many-evaluation,
  no-gradient competitor class; the "random restart" root of multi-start baselines.
- **Wake models:** Jensen/Park (Jensen 1983, Risø-M-2411; Katic, Højstrup, Jensen 1986,
  EWEC), **Bastankhah & Porté-Agel Gaussian (2014, Renew. Energy 70:116; yaw ext. 2016,
  JFM 806:506).** Smooth, C¹, analytically differentiable → well-behaved AEP gradients →
  gradient-magnitude-adaptive scheduling (`alpha0∝mean|∇AEP|`) is viable.
- **TopFarm / PyWake ecosystem:** TOPFARM (Réthoré et al. 2014, Wind Energy 17(12),
  10.1002/we.1667); **PyWake** (Pedersen et al., DTU; AD-capable AEP). The differentiable
  AEP + analytic gradients the skeleton consumes.
- **Multimodality reduction:** WEC (Thomas & Ning 2018/2022, above); **eight-method
  comparison (Thomas et al. 2023, WES 8:865)** — gradient methods dominate the top ranks
  (the baseline-framing evidence).
- **THE central reference — Quick et al. 2023 (WES 8:1235–1250, doi:10.5194/wes-8-1235-2023).**
  Monte-Carlo AEP-gradient SGD; the exact native `lr`/`alpha`/beta recipe (§7.8). *Citation
  provenance:* 2022 = WESD preprint `wes-2022-104`; **archival = 2023**, WES 8:1235–1250.
  Cite the archival version as **2023**.
- **Inclusion/exclusion zones — Criado Risco et al. 2024 (WES 9:585–600,
  doi:10.5194/wes-9-585-2024).** Analytical signed-distance-to-polygon (positive = inside
  inclusion / outside exclusion) + analytic gradients; ParqueFicticio case. **FunWake's
  exact boundary-constraint formulation.**
- **Reference turbines / benchmarks:** IEA 15 MW (Gaertner et al. 2020, NREL/TP-5000-75698,
  D=240 m — training scale), IEA 10 MW (Bortolotti et al. 2019, NREL/TP-5000-73492,
  D=198 m — held-out scale), Vestas V80 (D=80 m). IEA Task 37 blind comparison (Baker et
  al. 2019, AIAA 2019-0540) — top methods all gradient-based.
- **Multi-start / init:** best-of-N multi-start is the field norm for local optima; a good
  heuristic init can beat thousands of restarts (Valotta Rodrigues et al. 2024, WES 9:321).
  *Relevance:* the wind-aware grid init does part of a schedule's job; schedule gains
  compound with good seeding, and a discovered schedule must beat best-of-N.

---

## 9. LLM-driven discovery & quality-diversity (positioning)

- **FunSearch (Romera-Paredes et al. 2024, *Nature* 625:468, 10.1038/s41586-023-06924-6).**
  LLM + evaluator in an evolutionary loop with an **island model**; evolves a small
  "priority function." The direct ancestor — `schedule_fn` ≈ the evolved fragment.
- **AlphaEvolve (Novikov et al. 2025, arXiv:2506.13131).** Gemini ensemble + **evaluation
  cascade** + MAP-Elites/island DB, editing whole files by diff. The closest
  methodological match — FunWake's cascade evaluator = AlphaEvolve's cascade; multi-model
  mutation (Claude/Gemini/Codex) = its LLM ensemble. Differences: a narrow well-typed
  target + first-class QD archive + lineage/provenance.
- **ELM / OpenELM (Lehman et al. 2022, arXiv:2206.08896).** LLM as a far better *mutation
  operator* than random GP, curated by **MAP-Elites**. The theoretical justification for
  FunWake's core loop.
- **LLM hyper-heuristics:** EoH (Liu et al. 2024, ICML, arXiv:2401.02051 — NL "thought" +
  code, low query budget), ReEvo (Ye et al. 2024, NeurIPS, arXiv:2402.01145 — reflector
  gives *verbal gradients*), LLaMEA (van Stein & Bäck 2025, IEEE TEVC 29(2):331,
  arXiv:2405.20132 — evolves whole optimizers from error traces), Eureka (Ma et al. 2024,
  ICLR, arXiv:2310.12931 — evolves reward-fn code, "reward reflection"). *Relevance:*
  ReEvo/EoH/Eureka motivate feeding cascade scores back as *reflections* and storing the
  LLM's NL rationale in lineage; LLaMEA is the nearest "LLM evolves optimizer code on a
  black-box benchmark."
- **LLM-as-variation / optimizer primitives:** LMX (Meyerson et al. 2024, ACM TELO,
  arXiv:2302.12170 — few-shot crossover), OPRO (Yang et al. 2024, ICLR, arXiv:2309.03409 —
  optimization-by-prompting), **LLM-for-HPO (Zhang et al. 2023, arXiv:2312.04528 —
  matches/beats Bayesian opt; "model code as a hyperparameter").** *Relevance:* LMX → add a
  crossover operator over elites from distinct MAP-Elites cells; the HPO work is the direct
  precedent FunWake generalizes from static HPs to a *time-varying schedule program*.
- **QD foundations:** MAP-Elites (Mouret & Clune 2015, arXiv:1504.04909), Novelty Search
  (Lehman & Stanley 2011, Evol. Comput. 19(2):189), CVT-MAP-Elites (Vassiliades et al.
  2018, IEEE TEC 22(4):623, arXiv:1610.05729 — scales past 2–3 descriptors), QD survey
  (Pugh, Soros, Stanley 2016, Front. Robot. AI 3:40). QD×LLM: QDAIF (Bradley et al. 2024,
  arXiv:2310.13032), in-context QD (Lim et al. 2024, arXiv:2404.15794 — seed prompts from
  the archive).
- **Symbolic/evolutionary optimizer discovery (non-LLM ancestors):** AutoML-Zero (Real et
  al. 2020, ICML, arXiv:2003.03384 — evolves ML algorithms from primitives), **Lion /
  Symbolic Discovery of Optimization Algorithms (Chen et al. 2023, NeurIPS,
  arXiv:2302.06675 — program search finds a deployable optimizer)**, Neural Optimizer
  Search (Bello et al. 2017, ICML, arXiv:1709.07417 — RL controller emits update-rule +
  LR-schedule strings). *Relevance:* the strongest evidence that *searching program space
  for optimizer behavior* yields deployable, better-than-hand-tuned results — FunWake's
  thesis one level up (schedule vs update rule), via LLM+QD instead of RL/primitive GP.

---

## 10. Synthesis — design menu for `schedule_fn`

**Already embodied in the gen-0 seeds** (so the search starts here, not from scratch):
inverse-time/product lr decay + `α∝1/lr` + betas 0.1/0.2 (`native`, Quick 2023); cosine
annealing + exploratory bumps (`cosine` seed, SGDR); cyclic warm-restart lr + cyclic
alpha (`cyclic` seed). The frontier below is what the seeds do **not** yet embody — the
testable hypotheses.

| Output | Strongest untried ideas (→ source §) | One-line hypothesis |
|---|---|---|
| **lr** | linear / WSD decay to `gamma_min` (§6); one-cycle warmup→`c·D`→`gamma_min` (§2); short linear warmup (§3) | hold near `c·D`, then (near-)linear cool-down beats cosine/product decay |
| **alpha** | bounded plateau `α_max≈few·α₀` instead of `∝1/lr→∞` (§7.2/7.3); **delay** the ramp to mid-run (§7.3/7.5); **terminal feasibility spike** (§7.5); `α₀·(1+Ct)^p` dynamic penalty (§7.9) | a *decoupled, bounded, late* α is feasible AND better-conditioned than native's diverging coupling |
| **beta1** | anti-correlate with lr (one-cycle, §2); increasing (Sutskever) vs decaying (Demon) ramp (§4) | momentum as implicit ALM multiplier lets a *moderate* α enforce constraints |
| **beta2** | ramp up / RAdam-rectify (§4); AMSGrad-monotone late (§4) | ramping β2 tames early adaptive-lr variance and absorbs the `~α` constraint-curvature conditioning |
| **α↔`gamma_min`** | ε-constrained shrinking tolerance (§7.9) | schedule α so the enforced violation band contracts to `gamma_min` only at the end |

**Four highest-value, non-obvious bets (all from the penalty literature, which the model
under-volunteers):** (1) **bounded/logistic α plateau** (exact-penalty + ALM); (2)
**delayed α ramp** (LANCELOT/filter/graduated); (3) **terminal feasibility-restoration
spike** (filter/funnel); (4) **phase-transition the Adam moments with the α phase**
(β2↑/β1↓ in the feasibility phase). Each is a small, differentiable, open-loop
`schedule_fn` edit with a clear mechanism, and each is a clean ablation against the
native coupling.

---

## 11. Citation-confidence notes
- **Web-verified this compilation:** all arXiv ids, venues, and years in §§1–9 unless
  flagged below.
- **Canonical / non-arXiv (cited from established knowledge, not re-verified):** Robbins &
  Monro 1951; RMSprop (Tieleman & Hinton 2012 Coursera lecture); polynomial "poly" decay
  (a family, DeepLab/BERT usage, no single origin); the "Noam" name (community shorthand
  for the Vaswani et al. 2017 inverse-√ schedule); classic optimization texts
  (Nocedal & Wright, Bertsekas, Fiacco & McCormick) and the founding penalty/ALM/filter
  papers (Courant 1943; Hestenes/Powell 1969; Zangwill 1967; Fletcher & Leyffer 2002).
- **Flagged for a double-check before formal citation:** ELM full author list; the
  "LLM for Evolutionary Optimization" survey (arXiv:2509.08269); Cully & Demiris 2018
  page numbers; Réthoré et al. 2014 exact page range (DOI confirmed); Katic et al. 1986
  (EWEC, no DOI).
- **Quick et al. year:** cite the **2023** archival WES 8:1235–1250; "2022" is the WESD
  preprint only. (Note: the companion short paper's bibliography currently cites 2022 — if
  consistency across both documents is desired, reconcile to the 2023 archival version.)
