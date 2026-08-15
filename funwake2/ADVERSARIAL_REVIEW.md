Verified the three key un-verified candidates directly in source: the spacing 1% slack (`dei_layout.py:411`), the convex-hull boundary overwrite (`dei_layout.py:379-382` + check at 402), and the unsanitized `best['src']` in `_reflection()` (~L346) are all real code facts. Here is the report.

---

# FunWake-2 Adversarial Review

**Bottom line:** The core search fitness (`run_portfolio_explore._eff_score`, `min(score_c,-1)`) *does* close the exemplar gaming bug — an infeasible cell can no longer contribute its inflated AEP to the training search. That fix holds. **However, the same bug was re-introduced downstream in `select_deploy.py`, which is the code that actually chooses what to ship.** The deployment decision is gamed, selects the *worst* honest generalizer, and rests on a difference that is not statistically distinguishable from zero. These are the findings that threaten the scientific conclusion and should be fixed before any "beats native / generalist found" claim stands.

---

## 1. CONFIRMED issues (most severe first)

### 1.1 Deployment metric `fb_mean` rewards infeasible-cell AEP — the exemplar bug, re-introduced in the deploy selector
**`funwake2/select_deploy.py:55, 64, 84-85`** · gaming · **invalidates the deployment result**

`summarize()` builds `deltas = {k: r['delta_pct'] …}` with **no per-cell feasibility penalty**, and `fb_mean = fmean(deltas.values())` averages raw per-cell `delta_pct`. The gate (L84) only checks held-out ROWP feasibility, not the other cells feeding `fb_mean`. `run_portfolio_explore._eff_score` fixes exactly this with `min(score_c,-1)`; `select_deploy` does not.

**Failure scenario (reproduced from committed JSONs):** the deployed candidate `port190` is infeasible on `parque_n20` (4/5 seeds) and `dei_n50_uniform` (3/5). The infeasible `parque_n20` cell scores **+0.2732%** (turbines escaping the multizone into low-wake positions — the exact exemplar pathology) and contributes ~88% of the +0.0517% `fb_mean` numerator. Apply the `_eff_score` penalty to the same cells and `port190` flips to **−0.3277%** — it does *not* beat native, and fully-feasible `codex021` becomes the correct winner. The headline "only candidate whose mean beats native c*D" (L106) is an artifact of rewarding a non-deployable layout.

**Fix:** in `summarize()`, replace each cell's contribution with `min(delta_pct, -1)` when that cell's `cand_feasible` is not full (`n_feas != n_seeds`), reusing `_eff_score` semantics; or require `feas_cells == n_cells` for eligibility rather than only `held_out_feasible`.

---

### 1.2 Deploy pick uses `fb_mean` argmax, contradicting the documented PRIMARY=held-out metric — deploys the WORST held-out generalizer
**`funwake2/select_deploy.py:85` (docstring L8-12)** · methodology · **invalidates the deployment result**

Docstring: "PRIMARY = rowp_n74 held-out delta_pct … Selected = the feasibility-gated argmax of PRIMARY." Code: `eligible.sort(key=lambda r: r["fb_mean"], …)` then `w = eligible[0]`. `fb_mean` is the **in-sample** farm-balanced mean, not the held-out score.

**Failure scenario (from `state/validation/*.json`):** among the 5 gate-passers, ranked by the documented PRIMARY (held-out ROWP): iter109 (+0.0903%) > iter04 (+0.0764%) > codex021 (+0.0555%) > antigravity488 (+0.0525%) > **port190 (+0.0111%)**. The code deploys `port190` — the *worst* honest generalizer — because its `fb_mean` is inflated by the infeasible `parque_n20` cell (1.1). The one clean held-out signal the file claims to select on is discarded.

**Fix:** sort the deploy pick by held-out `ho['delta_pct']` (present `fb_mean` only as context), OR fix `fb_mean` per 1.1 and update the docstring; either way add a paired-uncertainty check (see 1.3).

---

### 1.3 Wins are within seed noise; best-of-N selected on the same cells used to validate (winner's curse, no uncertainty)
**`funwake2/select_deploy.py`; `run_validation.py`; `validate_freshproc.py`** · methodology · **invalidates the "beats native" claim**

Reported deltas are ~0.01–0.09% (≈0.5–5 GWh on ~5540 GWh) while per-seed AEP spread is several GWh. Validators compute only means (`cand_mean`, `native_mean`, `delta_pct`) — **no SE, CI, or paired test**, despite storing per-seed paired rows. `select_deploy` then takes an argmax over candidates (each already best-of-~190 search iterations) with no significance correction, and 5 of the 6 `fb_mean` cells are the search's own `DEFAULT_CELLS` (in-sample).

**Failure scenario:** deployed `port190`'s only clean held-out (ROWP) is +0.0111%, paired **t≈+0.59 (p≫0.05)** — indistinguishable from zero. Meanwhile iter109 has a *significant* held-out win (+0.0903%, t≈+4.12) but loses `fb_mean` only because it underperforms on `parque` cells it never trained on. The metric rewards in-sample fit and deploys the weakest honest generalizer.

**Fix:** compute per-cell and aggregate paired-difference CIs/bootstrap from the stored rows; require the held-out CI to exclude 0 before declaring a win; rank on out-of-portfolio held-out farms, not `fb_mean` over the search's training cells.

---

### 1.4 Non-atomic checkpoint write can brick a run on resume (infinite silent crash loop)
**`funwake2/run_portfolio_explore.py:375-383` (resume load L310); identical `run_codex_explore.py:250-256` (load L162)** · resume-state · does not invalidate results, but total-loss risk

`_ckpt()` does `json.dump(..., open(ckpt_path,"w"), indent=2)` — `open(...,"w")` truncates `summary.json` **before** writing, every iteration. The system is designed to be SIGKILL'd frequently (task-reaper in `sup_persist.sh`/`run_gen.sh`). A kill/OOM/reboot mid-dump leaves invalid JSON. On resume, `json.load(open(ckpt_path))` is **unguarded** → `JSONDecodeError` → process dies. `run_gen.sh`'s `while :` relaunches → crashes again forever; `count()` swallows the parse error as `n=0`, so the supervisor believes 0 attempts done and never stops. Trajectory + best are unrecoverable. `gbar_eval.py:76-78` already uses the correct `tmp + os.replace` pattern — it just wasn't applied here.

**Fix:** write to `ckpt_path+'.tmp'` then `os.replace(...)`; wrap the resume `json.load` in try/except with a `.bak` fallback.

---

### 1.5 `score_c` compares quadratic-penalty candidates against a frozen linear-penalty native baseline
**`funwake2/run_portfolio_explore.py:73` (also `run_validation.py:33`, `validate_freshproc.py:25`)** · baseline · **weakens reported %-over-baseline numbers, not candidate selection** (revised to medium)

`baselines_g2.json` native AEP values were generated 2026-07-31 under `_SPACING_QUADRATIC=False` (linear spacing penalty). The quadratic switch landed 2026-08-13 (`skeleton_v2.py:64`, commit a3916af7) and the baseline table was **never regenerated**. Every candidate now runs quadratic, so `score_c = 100*(cand_quad − nat_linear)/nat_linear` mixes constraint forms on spacing-active cells.

**Failure scenario:** re-running native fresh-quadratic vs the stored linear table gives a cell-dependent offset up to ~0.06–0.09% on spacing-active cells (parque_n20, parque_n10_omnidir, dei_n50_uniform) — a large fraction of the sub-0.1% reported deltas; for the tiny `dei_n50_uniform` delta this is up to an ~8x change / possible sign flip. **Mitigation (why medium):** the offset is near common-mode across candidates, so it does not materially reorder the search; its damage is to the *validity of reported/deployment %-over-baseline numbers*, which are exactly the numbers 1.1–1.3 already hinge on.

**Fix:** regenerate native baselines under the current constraint form after any skeleton change; stamp a constraint-form/version hash into the baseline JSON and assert it on load.

---

### 1.6 `FORBIDDEN_TOKENS` omits the primary held-out key `rowp_n74` — firewall guard has a hole for the actual holdout
**`funwake2/controller/workspace.py:41-46`** · firewall · **latent; leak only on gbar backend + operator misconfig** (revised to medium)

The forbidden-token list blocks `rowp_n200/rowp_n300/rowp_n74_uniform/problem_rowp/holdout` but **not the bare key `rowp_n74`** (`evaluator.py:160`, role=holdout). Matching is substring, and `rowp_n74_uniform` does not match the shorter `rowp_n74`; there is no bare `rowp` token. `_assert_feedback_firewalled` only rejects `aep`/`gwh`, so a feedback dict keyed `rowp_n74` (score% + feasibility) passes `assert_clean` and reaches the mutator prompt.

**Failure scenario / fail-closed caveats:** requires an operator to pass `--cells … rowp_n74` to the *search* (no shipped script does). On the local/Mac backend it **fails closed** — `_native()` KeyErrors because `rowp_n74` isn't in `baselines_g2.json`. Only the **gbar backend leaks** (its native baselines include `rowp_n74`, so no crash, and per-iteration holdout score+feasibility flows into `_build_prompt`).

**Fix:** add `rowp_n74` (and a bare `rowp` catch-all) to `FORBIDDEN_TOKENS`, and assert in `main()` that every `--cells` entry has evaluator `role=='train'`.

---

## 2. PLAUSIBLE issues worth a closer look (real code facts, not individually stress-tested to a full repro)

- **`_INFEAS_PENALTY = -1.0` is a fixed floor, not a strict dominator** — `run_portfolio_explore.py:64,69`. The comment claims it is "strictly worse than any realistic feasible deficit," but −1% is *not* worse than a −3% feasible deficit. On any cell whose best achievable *feasible* score is below −1%, the search would prefer to abandon feasibility there (capped at −1) over staying feasible. Not reachable on current portfolios (deficits < 1%), but it is the latent version of the exemplar bug and the comment is false. **Fix:** set the penalty far below the worst feasible score (e.g. −100) or make it proportional to the violation.

- **1% multiplicative spacing slack accepts real violations** — `benchmarks/dei_layout.py:411`, `spacing_ok = min_dist >= min_spacing*0.99`. **Confirmed present.** For DEI (min_spacing 960 m) that is a 9.6 m tolerance; a layout with closest pair 950.4–960 m is declared feasible and can be deployed while genuinely violating min-spacing — inconsistent with the multizone gate's 0.1 m absolute slack (`evaluator.py:242`). Given the scale-matched quadratic penalty's weak restoring force just below min_spacing, a schedule can bank slightly higher AEP by packing ~1% tighter. **Fix:** use an absolute tolerance, `min_dist >= min_spacing - 0.1`.

- **Boundary check convex-hulls the polygon** — `benchmarks/dei_layout.py:379-382`, then `check_feasibility` (L402) and the optimizer penalty (`skeleton_v2.py:209`) use `convex=True`. **Confirmed present.** All currently-registered single-poly cells are convex, so latent today — but the framework explicitly targets generalization to unseen/irregular farms; a concave (L-shaped/notched) farm would let a turbine sit in a concavity, *outside* the true polygon but inside the hull, and be scored feasible. The concave-correct path (`polygon_sdf`, `convex=False`) already exists and is used for multizone. **Fix:** don't convex-hull; use `convex=False` for single-poly, or assert convexity at load.

- **Reflection channel embeds `best['src']` UNSANITIZED into the prompt** — `run_portfolio_explore.py:~346`. **Confirmed present.** `parent_source` goes through `W.sanitize()` (which strips docstrings / redacts forbidden tokens), but `_reflection()` splices `best['src']` verbatim into a code fence in `ctx.notes`, and `_build_prompt` inserts `notes` with no sanitize; `assert_clean` only scans files, never the prompt string. Seeding from a source whose docstring carries provenance (e.g. `native.py`'s `results/…`, `lr0_…` — themselves forbidden tokens) would show the mutator exactly the design provenance sanitize exists to hide. **Fix:** wrap the embedded source in `W.sanitize(best['src'])`.

- **gbar native baseline drops its `feasible` flag** — `run_portfolio_explore.py:177`, `_fetch_gbar_baselines` reads only `aep`. On knife-edge cells (native `dei_n50_uniform` sits at min_dist 959.99 vs 960) cross-platform drift could flip the native infeasible on gbar while the Mac table shows feasible; its (inflated) AEP still becomes the `score_c` denominator with no warning. Candidates get an infeasibility penalty; the native reference gets none. **Fix:** carry per-seed `feasible` through and refuse/flag any native infeasible on a requested seed.

---

## 3. REFUTED / non-issues (checked, safe)

- **"`feas_slack>0` lets the search pick an infeasible `best`" (`run_portfolio_explore.py:439`)** — **Refuted for HEAD.** `_eff_score` (`min(score_c,-1)`) feeds the very `pct` that best-selection compares, so tipping a farm infeasible *loses* `fb_mean`. `feas_slack` only gates eligibility and bites only when no all-feasible candidate exists yet (its documented degenerate-portfolio purpose), where hill-climbing still drives toward feasibility. The cited `explore_scale14/summary.json` "confirmation" is **stale** — committed at an ancestor of the commit that added `_eff_score`, and arithmetically impossible under current code. Residual: the checkpoint field is mislabeled `best_feasible` (cosmetic).

- **"Portfolios include cells with infeasible native c*D baseline" (`run_scaleN_loop.sh:13`)** — **Refuted as a gaming/contamination vector.** True that `parque_n30_uniform`/`dei_n100` have no all-feasible native, but such a cell contributes a constant −1 (`_eff_score`) to every candidate — a fixed offset that cancels in the argmax; the inflated native AEP never enters best-selection. The "gate unsatisfiable → feas_slack forced on" state is the documented capability-frontier design (registry already marks these `stage_b=False`/`feasibility_only`), not an artifact.

Both refuted findings are the *same mechanism* as the exemplar; the honest reading is that `_eff_score` genuinely closes it **in the training search**. The live danger is that `select_deploy` (Section 1) never got the same fix.

---

## 4. Un-verified candidates worth a manual look

- **Inconsistent seed counts / runners in `select_deploy` cross-ranking** (`select_deploy.py:72`) — `iter04` validates ROWP with 5 seeds while others use 10; `port190` was scored by `validate_freshproc.py` (fresh-process, because it190 leaks a cached tracer under in-process reuse) while the rest used in-process `run_validation.py`. At 0.02%-scale margins this is apples-to-oranges. Worth enforcing one runner + fixed seed set and asserting provenance per JSON before ranking. (Compounds 1.3.)

- **Resume rebuilds the pool with a strict all-feasible filter** (`run_portfolio_explore.py:312`, `_mk` L298-307) — live eligibility is `n_feas_cells >= len(cells)-feas_slack`, but trajectory stores only all-cells-`feasible`, and resume rebuilds `pool = [t for t in traj if t["feasible"]]` (slack effectively 0), then `_mk` hardcodes `feas=True`/`viol=0` per restored parent. On a `feas_slack` portfolio the resumed pool can collapse to `best+native`, and post-resume `_note`/`_fb` tell the mutator every cell is feasible — erasing the "fix feasibility on <cell>" signal. Given frequent task-reaper restarts, worth confirming whether this degrades real runs.

- **Eval timeouts counted against the attempt budget** (`run_portfolio_explore.py:421-424`) — `except Exception` records `{"status":"eval-error"}` which still increments `len(traj)` toward `--iters` and the supervisor `count()`, despite the docstring's "budget = REAL scored attempts." A flaky gbar worker would end a search early with a truncated `best`, and dispatch-timing (not quality) decides which candidates survive. Worth gating the while-loop on scored attempts and distinguishing infra timeouts from candidate crashes.

- **`feasibility_only` cells not excluded from the %-mean** (`run_portfolio_explore.py:111,154`) — `_score`/`_aggregate` fold every cell into `fb_mean` with no check of the `feasibility_only` flag; harmless only because the default portfolio omits `parque_n14_uniform`. Low priority.

---

### Priority for the author
Fix **1.1 + 1.2 + 1.3 together** — they are one defect (the deploy selector rewards ungated infeasible in-sample AEP and ignores the held-out primary and uncertainty), and together they mean the current "portfolio it190 beats native / generalist found" conclusion is not supported: the correct feasibility-penalized, significance-aware selection deploys a different (fully-feasible) candidate and finds no candidate that significantly beats native on the clean held-out farm. **1.5** must be fixed before any %-over-baseline number is trusted. **1.4** and **1.6** are robustness/firewall hardening. The training-search fitness itself is sound.