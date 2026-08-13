> **Provenance:** Produced by an 8-agent literature-review workflow (6 parallel domain
> surveys with live web search → synthesis → adversarial novelty check). Citations were
> instructed to be real-only. The critique's two headline papers (EvoStage, Celo) were
> independently verified against arXiv/web by the author before inclusion. Individual
> recent (2024–2026) citations should still be citation-checked before use in a paper.

---

## ⚠ Primary-source correction (EvoStage) — supersedes the claims below

After the workflow, the EvoStage paper (Lu, Xue, Gao, Shi, Xu, Yuan, Qian & Zhou,
arXiv:2603.07970, Mar 2026) was read in full (§3 + Figs. 3–4). It confirms and
**strengthens** the scooping finding, and corrects two errors made below:

- **EvoStage designs the learning-rate schedule** for the Adam optimizer in chip
  global placement (§3 title; "design the learning rate schedule for the Adam
  optimizer"; the designed object is a "strategy function" of the optimization state
  — decay factor adjusted on hpwl/overflow/grad-norm, Figs. 3–4). Confirmed.
- **EvoStage ALSO designs the constraint-PENALTY schedule.** For the commercial 3D
  tool it evolves "the learning rate schedules **as well as the density weight
  schedules**"; the density weight is the Lagrangian/penalty coefficient λ in
  `min Σ WL + λ·D(x,y)` s.t. density ≤ dₜ. **This invalidates the claim below (and in
  the appendix) that a coupled feasibility/penalty co-schedule is FunWake-2's
  distinctive contribution — it is not; EvoStage anticipates it.** The substrate
  (minimize wirelength s.t. a density constraint, via increasing-penalty Lagrangian
  relaxation) is structurally the same constrained-layout problem shape as WFLO.

**Re-centered novelty (what EvoStage does NOT do):** the schedule-discovery *paradigm*
and the *penalty co-scheduling* are both anticipated. FunWake-2's genuinely novel,
un-scooped core is therefore narrower and lies in the **generalization science**, not
the method:
1. **Cross-instance / axis-coverage generalization study** — specialists → farm-balanced
   *portfolio* training → the finding that transfer is limited to farm-characteristic
   axes *covered* in training (wind-rose directionality, turbine count N) → leave-one-
   farm-out CV → deployment selection by a farm-balanced-mean metric. EvoStage has no
   analogue: it reports SOTA per chip case, optimizing each instance, with no held-out /
   distribution-transfer framing.
2. **Scale-aware descriptor conditioning for one-schedule-many-instances** — the schedule
   is a function of `(D, N, min-spacing, alpha0)`, so a *single* discovered schedule
   transfers across farms; EvoStage re-designs per case.
3. **Mechanism difference** — EvoStage's schedule is closed-loop/adaptive on runtime
   state with stagewise feedback; FunWake-2's is a mostly open-loop function of step +
   static farm descriptors.

Frame the writeup around the **generalization study + WFLO domain transfer**, cite
EvoStage as the closest concurrent *method*, and do **not** claim the LLM-schedule-
discovery paradigm or penalty co-scheduling as novel.

---

# Positioning FunWake-2: A Literature Review

FunWake-2 uses LLM agents (Claude Opus, a GPT-5-class "codex", and Gemini-3-Pro via "antigravity") in an evolutionary loop to write and mutate the **learning-rate and constraint-penalty schedule** — a function returning `(lr, alpha, beta1, beta2)` per step — for a *fixed* gradient-based (Adam / TopFarm-SGD) wind-farm layout optimizer. Candidates are scored by actually running ~8000 optimizer steps on wind-farm problems and measuring Annual Energy Production (AEP) under boundary and inter-turbine spacing feasibility. This review situates that contribution against five adjacent bodies of work.

## 1. Taxonomy of adjacent fields

FunWake-2 sits at the intersection of five literatures:

1. **LLM-driven algorithm discovery** — LLMs as evolutionary mutation operators over executable code (FunSearch, EoH, ReEvo, LLaMEA, Eureka, FunBO, AlphaEvolve). *Relation:* FunWake-2 inherits this loop but targets an optimizer's internal schedule rather than a standalone heuristic.
2. **Learned optimizers (L2O)** — searching or meta-learning optimizer behavior (Andrychowicz, Metz/VeLO, Lion). *Relation:* FunWake-2 shares the goal of discovered optimization behavior but fixes the update rule and evolves only the schedule wrapped around it, under hard constraints.
3. **Schedule / HPO / AutoML** — automated discovery of learning-rate and hyperparameter schedules (AutoLR, PBT, SGDR, one-cycle, WSD, hypergradient). *Relation:* FunWake-2 evolves the schedule *shape* but adds a coupled penalty weight and cross-problem transfer.
4. **Wind-farm layout optimization (WFLO)** — the physical solver, baseline, and objective (Valotta Rodrigues, Thomas, Shin). *Relation:* This is the substrate FunWake-2 operates on.
5. **Distribution / size generalization and meta-validation** — when learned solvers transfer (Joshi, Setlur, Amos, DHEvo, EvoTune, VeLO, Yao, No Free Lunch). *Relation:* This frames FunWake-2's central empirical claims.

## 2. Per-field relation

### LLM-driven algorithm discovery

The foundational template is **FunSearch** (Romera-Paredes et al., 2024): an LLM paired with a programmatic evaluator in an island-evolution loop, discovering cap-set and bin-packing heuristics. FunWake-2 inherits this exact loop but applies it to an inner optimizer's schedule. **EoH** (Liu et al., 2024) and **ReEvo** (Ye et al., 2024) extend the paradigm with natural-language "thoughts" and reflective verbal gradients over combinatorial problems; FunWake-2 shares the LLM-as-mutation-operator design but targets a continuous constrained optimizer rather than discrete-COP heuristics. **LLaMEA** (van Stein & Bäck, 2024) is the nearest method-side analogue for continuous problems, but it evolves *whole metaheuristics* for BBOB benchmarks, whereas FunWake-2 evolves only the `(lr, alpha, beta1, beta2)` schedule of a fixed Adam/SGD skeleton. **Eureka** (Ma et al., 2023) evolves RL reward code — a control signal shaping an inner optimization, closely analogous to shaping an inner AEP optimizer, but the object (reward vs. schedule) and domain differ. **AlphaEvolve** (Novikov et al., 2025) mirrors FunWake-2's multi-LLM code-improvement setup at codebase scale; FunWake-2 is a narrow, generalization-focused instance. **OPRO** (Yang et al., 2023) represents the "LLM directly *is* the optimizer" branch — the opposite of FunWake-2, where the LLM is a designer of a fixed gradient optimizer. **ADAS** (Hu, Lu & Clune, 2024) and the **LLM4AD** survey (Liu et al., 2025) position FunWake-2 in the "LLM-as-Designer + evolutionary-search" cell.

**Closest here:** **FunBO** (Aglietti et al., 2024) — the only work that uses LLM-driven evolution to discover *a component of an optimizer* (the Bayesian-optimization acquisition function) as executable code, scored by running the optimizer, and makes in- vs. out-of-distribution generalization the central axis. FunWake-2 differs by targeting the per-step schedule of a *constrained* gradient solver and by giving a structured (axis-coverage) account of transfer rather than a single in/out split.

### Learned optimizers (L2O)

**Andrychowicz et al. (2016)** originated L2O — learning the update rule via backprop through the optimizee. **Metz et al. (2019)** documented why gradient-based meta-training is unstable (biased/exploding truncated-unroll gradients), which motivates FunWake-2's gradient-free route: LLM mutation plus direct scoring sidesteps unrolling entirely. **Metz et al. (2020)** and **VeLO** (Metz et al., 2022) established that *task diversity* drives out-of-distribution generalization, meta-training at massive scale (~4000 TPU-months for VeLO). FunWake-2 re-derives the diversity principle as its farm-balanced portfolio result but at negligible compute, and reports where versatility still breaks. The **Learning-to-Optimize primer** (Chen et al., 2022) frames FunWake-2's core claim that searched optimizers overfit their training distribution.

**Closest here:** **Lion** (Chen et al., 2023) — gradient-free evolutionary/symbolic program search for an optimizer, tested for cross-architecture generalization; structurally the same mutate-run-score-select loop. FunWake-2 differs on three axes: it mutates a *schedule* not the *update rule*; the operator is an LLM writing code, not regularized evolution over a symbolic grammar; and fitness is *feasibility-gated* on a constrained physical objective, not a smooth training loss.

### Schedule / HPO / AutoML

**AutoLR** (Carvalho et al., 2020) is the closest classical analogue: Structured Grammatical Evolution over LR-schedule *functions*, scored by actually training, beating hand-tuned baselines and rediscovering known shapes. **Morgan & Hougen (2024)** jointly evolve the update equation, decay functions, and LR schedule. Both differ from FunWake-2 on the three things that define it together: LLM (not GP-grammar) mutation, a coupled *penalty weight* (`alpha`) for feasibility rather than LR alone, and a *scale-aware* schedule conditioned on problem descriptors for cross-*problem* transfer. **PBT** (Jaderberg et al., 2017) is the canonical origin of discovering a hyperparameter *schedule* by population evolution; FunWake-2 replaces random mutation with LLM-authored code and studies transfer to unseen problems. **SGDR** (Loshchilov & Hutter, 2017), **one-cycle super-convergence** (Smith & Topin, 2018), and **WSD** (Wen et al., 2024) are hand-designed schedule shapes FunWake-2's agents can rediscover or beat — notably, one-cycle's momentum/LR coupling echoes FunWake-2's discovered (rather than prescribed) alpha-vs-LR coupling. **Hypergradient descent** (Baydin et al., 2017) is the analytic non-search alternative; FunWake-2 instead searches an explicit closed-form schedule with descriptor conditioning rather than per-step feedback. **DiscoPOP** (Lu et al., 2024) is the nearest methodological twin for LLM-in-the-loop code evolution but targets a training *loss* for LLM alignment, without scale-aware cross-problem generalization.

### Wind-farm layout optimization

**Valotta Rodrigues et al. (2024)** *is* the exact substrate: TopFarm/PyWake gradient-based (AD) WFLO with multi-start SGD, boundary + spacing constraints, and a Smart-Start (~c·D scale) heuristic init, with iterations scaling ~2.3·N and the alpha~1/lr coupling. Crucially, that paper *hand-designs* this schedule as a fixed physics-motivated recipe; FunWake-2 has LLM agents *evolutionarily search* it and studies cross-farm transfer. **Thomas et al. (2023)** benchmarks eight WFLO methods on IEA-Task-37-style farms (10 MW / 198 m rotor — the class of the held-out ROWP test), finding method-agnostic wake-loss gains, which motivates FunWake-2's focus on the *schedule* rather than solver choice. **Shin et al. (2025)** apply diffusion models to generate layouts *directly*; FunWake-2 instead generates the *optimizer's schedule* that then produces layouts — a schedule-discovery vs. layout-generation distinction.

### Distribution / size generalization and meta-validation

**Joshi et al. (2021)** established the pathology FunWake-2 confronts on the turbine-count axis: learned solvers overfit trained scale and fail out-of-distribution unless training covers the target regime. **LEHD** (Luo et al., 2023) pursues scale generalization by architecting a solver; FunWake-2 instead makes scale-awareness explicit via rotor-diameter/N conditioning. **Setlur et al. (2021)** show measured generalization depends on training-distribution diversity and protocol — directly paralleling FunWake-2's specialist-vs-portfolio finding and leave-one-farm-out selection. **Amos (2023)** provides the amortized-optimization framing: FunWake-2 amortizes a *schedule* over a distribution of farm problems keyed to descriptors. **Yao et al. (2023)** and **No Free Lunch** (Wolpert & Macready, 1997) supply the essential caveats: raw diversity is not automatically sufficient, and any advantage must come from restricting to a structured problem class — the theoretical root of the axis-coverage finding.

**Closest here:** **EvoTune** (Surina et al., 2025) — LLMs as mutation operators in an evolutionary loop scored by execution, with explicit generalization tests on perturbed and held-out instances, mirroring FunWake-2's execute-to-score search plus held-out/LOFO testing. **DHEvo** (Zhang et al., 2024) is a close second on the *thesis* (LLM-evolved optimizer components overfit and need distribution-aware training), which parallels FunWake-2's portfolio remedy.

## 3. The closest individual works overall

1. **FunBO** (Aglietti et al., 2024) — differs in that FunWake-2 discovers the per-step lr/penalty *schedule* of a *constrained* gradient optimizer (not an unconstrained BO acquisition function) and characterizes transfer as axis-coverage-limited rather than a single in/out split.
2. **Lion / Symbolic Discovery** (Chen et al., 2023) — differs in that FunWake-2 uses LLM agents (not symbolic grammar evolution) to mutate a *schedule* (not the update rule) for a *feasibility-gated* physical objective.
3. **AutoLR** (Carvalho et al., 2020) — differs in that FunWake-2 uses LLM (not GP) mutation, co-schedules a constraint-penalty `alpha` (not LR alone), and studies cross-*problem* (not per-architecture) generalization.
4. **Valotta Rodrigues et al. (2024)** — differs in that FunWake-2 *searches* the lr/alpha coupling this paper *hand-designs*, and studies its cross-farm generalization rather than treating it as a fixed recipe.
5. **EvoTune** (Surina et al., 2025) — differs in that FunWake-2's search object is a continuous lr+penalty schedule for a *constrained physics* optimizer, with leave-one-farm-out validation, not discrete-program discovery.
6. **DHEvo** (Zhang et al., 2024) — differs in that FunWake-2 cures overfitting via a *farm-balanced portfolio* over physical descriptor axes (not co-evolved MILP instances) and gives an explicit coverage account of *which* axes transfer.

## 4. Distinctive contributions

Honestly stated, FunWake-2's novelty is a *combination* no single prior work holds:

1. **LLM/evolutionary discovery of the lr+constraint-penalty SCHEDULE** — not weights (unlike Andrychowicz/VeLO), not the update rule (unlike Lion), not the whole metaheuristic (unlike LLaMEA), but the `(lr, alpha, beta1, beta2)` annealing wrapped around a *fixed* constrained solver.
2. **A coupled feasibility penalty** — the `alpha`-vs-`lr` co-schedule handles hard boundary + spacing constraints, absent from AutoLR/PBT/SGDR/one-cycle/WSD, which all schedule only LR/momentum on unconstrained losses.
3. **Scale-aware parameterization** — the exploration LR is built from rotor diameter D and conditioned on N, min-spacing, and gradient magnitude, so one schedule can transfer across *different* farm instances (unlike architecture-bound AutoLR/Morgan).
4. **The axis-coverage generalization finding** — schedules transfer on farm-characteristic axes *covered* in training (wind-rose directionality, turbine count N) and fail out-of-distribution on uncovered axes — a sharper, mechanistic account than "more diversity helps" (Metz) or "meta-overfitting happens" (Setlur).
5. **Surpassing the physics baseline in-regime** — a discovered schedule can beat the physics-motivated c·D baseline on covered regimes, even reaching feasibility on unidirectional-wind farms where the base solver cannot — while that baseline remains the robust default out-of-distribution.
6. **Deployment selection via held-out + leave-one-farm-out CV** and a farm-balanced-mean metric, rather than a single held-out farm.

## 5. Honest limitations

Several claims should be tempered. **The generalization principle is not new** — that a diverse training distribution buys transfer is the central result of Metz et al. (2020) and VeLO, and Setlur et al. (2021) and Yao et al. (2023) already show diversity is not automatically sufficient; FunWake-2's contribution is a concrete instantiation and a sharper *axis-coverage* framing, not a new mechanism, and No Free Lunch (Wolpert & Macready, 1997) guarantees such coverage limits exist a priori. **Scale is modest** — VeLO meta-trains across thousands of tasks; FunWake-2 covers a small portfolio of farms, so its transfer claims rest on far fewer axes than the L2O literature validates, and "beats the c·D baseline" holds only in-regime, with that baseline explicitly the more robust default out-of-distribution. **The search space is narrow** — fixing the Adam/TopFarm-SGD skeleton means FunWake-2 cannot discover genuinely new optimizer behavior the way Lion or LLaMEA can; it tunes a recipe rather than inventing one. **Method rigor** — unlike EvoTune (which couples evolution with RL/DPO to improve the generator) or ReEvo's structured reflection, FunWake-2's LLM mutation is comparatively ad hoc, and the multi-LLM ensemble (Opus/codex/antigravity) is not ablated against a single agent. **The WFLO gains are method-agnostic** — Thomas et al. (2023) show very different solvers reach similar wake-loss reductions, so schedule discovery may capture only a small slice of achievable AEP. Finally, the held-out ROWP evaluation is a single test farm; robust deployment claims would require the broader farm family that VeLO-style meta-generalization studies use.
---

# Appendix — Adversarial Novelty Assessment

*A dedicated critic agent was tasked with **refuting** FunWake-2's novelty by finding
prior work that already did it. Its two headline citations were verified real: **EvoStage**
(Lu, Xue, Gao, Shi, Xu, Yuan, Qian & Zhou — "Advancing Automated Algorithm Design via
Evolutionary Stagewise Design with LLMs") and **Celo** (Moudgil, Knyazev, Lajoie,
Belilovsky — "Training Versatile Learned Optimizers on a Compute Diet").*

I have enough to deliver the adversarial assessment. Here are my findings.

## Adversarial stress-test of FunWake-2's novelty

### (a) Single closest prior work — the real scooping risk

**EvoStage** — "Advancing Automated Algorithm Design via Evolutionary Stagewise Design with LLMs," Lu, Xue, Gao, Shi, Xu, Yuan, Qian & Zhou (arXiv:2603.07970, 2026). **The review does not cite it, and it is the closest work by a wide margin — closer than FunBO, Lion, AutoLR, or EvoTune.**

Overlap, point by point against FunWake-2's defining claims:
- **Same loop**: LLM multi-agent system used as the evolutionary mutation operator, with real-time execution feedback.
- **Same search object**: it evolves the **learning-rate schedule of the Adam optimizer** *and* the **optimization-step schedule for a sequence of Lagrangian-relaxation subproblems** — i.e. a per-stage learning-rate + constraint-handling schedule wrapped around a *fixed* gradient solver. That is nearly identical to FunWake-2's `(lr, alpha, …)` schedule over a fixed Adam/SGD skeleton. The Lagrangian-relaxation step schedule is functionally a **penalty/constraint schedule**, which the review claims as its distinctive contribution #2.
- **Same substrate class**: chip placement — a **constrained physical layout optimization** minimizing half-perimeter wire-length (HPWL) subject to cell **density/overlap constraints** `D(x,y) ≤ d_t`. This is structurally the same problem shape as WFLO (maximize AEP subject to boundary + spacing feasibility): a continuous constrained layout objective with a penalty coupling.
- **Same fitness**: candidates scored by **actually running the placement optimizer** on real benchmarks (ISPD 2005, ICCAD 2015; 16 chip cases).

Where FunWake-2 still differs (genuine, but narrower than the review implies): (i) it evolves an **explicit per-step penalty weight `alpha` co-scheduled with lr and the Adam betas**, whereas EvoStage schedules lr + subproblem-step counts; (ii) it makes **cross-instance / axis-coverage generalization the central object** — EvoStage reports per-case wins but does *not* test cross-benchmark transfer (e.g. train ISPD → test ICCAD); (iii) FunWake-2's **scale-aware descriptor conditioning** (rotor D, N, min-spacing, grad magnitude). EvoStage may also be roughly **concurrent** (arXiv March 2026), so it is a concurrent-work / scooping risk rather than clearly-predating prior art — but it substantially anticipates the *paradigm*.

### (b) Overstated novelty claims that should be softened

1. **"FunBO … the only work that uses LLM-driven evolution to discover a component of an optimizer"** (Section 2). **False.** EvoStage discovers optimizer *schedule* components via LLM evolution; DiscoPOP discovers a loss. Remove "only."
2. **Contribution #1 — "a combination no single prior work holds."** Overstated. EvoStage holds most of the bundle (LLM-evolved lr schedule + Lagrangian/penalty step schedule + fixed constrained-layout solver + execute-to-score). The honest residual is the *explicit per-step alpha co-schedule* and the *generalization framing*, not the paradigm.
3. **Contribution #2 — coupled feasibility penalty "absent from AutoLR/PBT/SGDR/one-cycle/WSD."** True for those specific schedule papers, but the framing implies scheduling a constraint-penalty is new in general. It is not: **penalty / augmented-Lagrangian continuation** (increasing penalty weight along a schedule) is textbook constrained optimization (Nocedal & Wright), and EvoStage schedules exactly a Lagrangian-relaxation sequence. Reframe as "learned/evolved per-step penalty coupling," not "penalty scheduling is novel."
4. **Contribution #4 — axis-coverage as "a sharper, mechanistic account."** Overstated: it is an empirical observation on two axes (wind directionality, N), not a mechanism. The VeLO/Celo line already characterizes *which* task properties drive transfer. Downgrade "mechanistic" to "a concrete, domain-specific instantiation."
5. **Limitations lean on "VeLO needs 4000 TPU-months."** That framing is now contested — see Celo and Rezk et al. below — so "diversity requires massive compute" should not be stated as settled.

### (c) Missing citations (add these)

1. **EvoStage** — Lu et al., arXiv:2603.07970 (2026). *Mandatory* — the closest work; must be cited and distinguished (see (a)).
2. **Celo: Training Versatile Learned Optimizers on a Compute Diet** — arXiv:2501.12670 (2025). Achieves strong OOD meta-generalization in ~24 GPU-hours; directly undercuts the review's "modest scale vs. VeLO's 4000 TPU-months" framing.
3. **"Is Scaling Learned Optimizers Worth It? Evaluating the Value of VeLO's 4000 TPU Months"** — Rezk et al., arXiv:2310.18191 (2023). Important caveat on the VeLO generalization claims the review relies on.
4. **AgentHPO / "Large Language Model Agent for Hyper-Parameter Optimization"** — Liu et al., arXiv:2402.01881 (2024), and follow-up arXiv:2506.15167 (2025). The LLM-agent-for-optimizer-configuration branch is directly adjacent and currently absent.
5. **"Evolving Deep Learning Optimizers"** — arXiv:2512.11853 (2025). Recent evolutionary optimizer-discovery work worth positioning against.
6. **Nocedal & Wright, *Numerical Optimization*** (penalty / augmented-Lagrangian continuation) — to honestly ground the claim that scheduling a constraint-penalty weight has classical precedent.

### (d) Verdict

**Incremental / partly already-done — not a novel paradigm.** FunWake-2's headline contribution (LLM-evolutionary discovery of the lr + constraint-penalty schedule of a *fixed constrained physical-layout optimizer*, scored by execution) is substantially anticipated by EvoStage (concurrent). What survives as genuinely novel is narrower but real: the **explicit per-step `alpha`-vs-`lr` co-schedule with Adam betas**, the **scale-aware descriptor conditioning for cross-farm transfer**, and the **axis-coverage generalization study with leave-one-farm-out selection in the WFLO domain**. The review should be rewritten to cite EvoStage as the closest work, drop "only"/"no single prior work holds" phrasings, and reposition novelty as *domain transfer + explicit penalty co-scheduling + a generalization-axis analysis* rather than a new discovery paradigm.

Sources: [EvoStage](https://arxiv.org/abs/2603.07970), [FunBO](https://arxiv.org/abs/2406.04824), [Lion / Symbolic Discovery](https://arxiv.org/pdf/2302.06675), [Celo](https://arxiv.org/html/2501.12670), [Is Scaling Learned Optimizers Worth It?](https://arxiv.org/pdf/2310.18191), [AgentHPO](https://arxiv.org/abs/2402.01881), [Neural Optimizer/LR-schedule Joint Evolution](https://arxiv.org/pdf/2404.06679), [Evolving Deep Learning Optimizers](https://arxiv.org/pdf/2512.11853).