"""Run the Codex CLI as a FunWake-2 mutation engine and see what schedule it
discovers. EXPLORATORY: scores only on the DEI training cell (dei_n50) that an
agent is allowed to see — it does NOT touch the ROWP holdout or the pre-registered
test set, so it does not front-run the deployment decision.

Loop: sanitize the parent -> Codex proposes an improved schedule_fn (scoped, read-
only) -> score it on dei_n50 with the REAL evaluator -> feed the %-over-baseline
back -> hill-climb on the best feasible candidate. Prints the trajectory and the
best schedule Codex produced.

  pixi run python funwake2/run_codex_explore.py --iters 3 --seeds 0 1 2
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for _p in (_ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluator as E                                            # noqa: E402
from funwake2.controller import workspace as W                  # noqa: E402
from funwake2.controller.engines.base import EvoContext         # noqa: E402
from funwake2.controller.engines.codex_cli import CodexCLIEngine  # noqa: E402
from funwake2.controller.engines.claude_cli import ClaudeCLIEngine  # noqa: E402
from funwake2.controller.engines.gemini_cli import GeminiCLIEngine  # noqa: E402

_ENGINES = {"codex": (CodexCLIEngine, "gpt-5.5"),
            "claude": (ClaudeCLIEngine, "claude-opus-4-8"),
            "gemini": (GeminiCLIEngine, "gemini-2.5-flash")}

OUT = os.path.join(_THIS, "state", "codex_explore")
CELL = "dei_n50"


def _baseline(seeds):
    b = json.load(open(os.path.join(_THIS, "controller", "baselines_g2.json")))
    cb = b["cells"][CELL]["seeds"]
    return {s: cb[str(s)] for s in seeds}


def _prior_art():
    """Extract the §10 'Synthesis — design menu' block from the frozen prior-art
    survey (firewall-safe) to inject as static context into the codex prompt."""
    import re
    p = os.path.join(_THIS, "PRIOR_ART.md")
    if not os.path.exists(p):
        return ""
    m = re.search(r"## 10\. Synthesis.*?(?=\n## 11\.)", open(p).read(), re.S)
    return m.group(0).strip() if m else ""


def _valid(src):
    import ast
    try:
        t = ast.parse(src)
    except SyntaxError:
        return False
    return any(getattr(n, "name", "") == "schedule_fn" for n in ast.walk(t))


def _score(src, seeds, steps):
    path = os.path.join(OUT, "_cand.py")
    with open(path, "w") as f:
        f.write(src)
    fn = E.load_schedule(path)
    recs = [E.evaluate(CELL, fn, seed=s, total_steps=steps, gamma_min=0.01) for s in seeds]
    aeps = [r["aep_gwh"] for r in recs]
    feas = all(r["feasible"] for r in recs)
    ms = float(recs[0].get("min_spacing") or 0.0)
    max_bnd = max(float(r.get("boundary_penalty", 0.0) or 0.0) for r in recs)
    min_dist = min(float(r.get("min_dist_m", 1e9) or 1e9) for r in recs)
    spacing_short = max(0.0, (ms - min_dist) / ms) if ms else 0.0
    viol = 0.0 if feas else (max_bnd + spacing_short)   # 0 iff feasible
    return {"aep": statistics.fmean(aeps), "feas": feas,
            "n_feas": sum(1 for r in recs if r["feasible"]),
            "max_bnd": max_bnd, "min_dist": min_dist, "min_spacing": ms, "viol": viol}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--engine", choices=list(_ENGINES), default="codex")
    ap.add_argument("--model", default=None, help="override engine default model")
    ap.add_argument("--tag", default=None, help="output subdir (default = engine)")
    ap.add_argument("--seed-from", dest="seed_from", default=None,
                    help="start the loop from this schedule file (continue), not native")
    ap.add_argument("--resume", action="store_true", help="resume from OUT/summary.json")
    ap.add_argument("--pool-k", dest="pool_k", type=int, default=6,
                    help="keep top-K feasible schedules for diverse parent sampling")
    ap.add_argument("--restart-every", dest="restart_every", type=int, default=12,
                    help="every Nth attempt, restart the parent from native (explore)")
    ap.add_argument("--prior-art", dest="prior_art", action="store_true", default=True,
                    help="inject the frozen PRIOR_ART.md §10 design menu (default on)")
    ap.add_argument("--no-prior-art", dest="prior_art", action="store_false")
    ap.add_argument("--reflect", action="store_true", default=True,
                    help="reflection channel: show best-so-far + recent failed deltas (default on)")
    ap.add_argument("--no-reflect", dest="reflect", action="store_false")
    args = ap.parse_args()

    global OUT
    OUT = os.path.join(_THIS, "state", "explore_" + (args.tag or args.engine))
    os.makedirs(OUT, exist_ok=True)
    EngCls, default_model = _ENGINES[args.engine]
    model = args.model or default_model

    base = _baseline(args.seeds)
    base_mean = statistics.fmean(base.values())
    print(f"[{args.engine}] cell={CELL} baseline(native c*D) mean AEP over seeds "
          f"{args.seeds} = {base_mean:.4f} GWh", flush=True)

    scope = os.path.join(OUT, "scope")
    eng = EngCls(model=model, cwd=scope)
    eng.preflight()
    print(f"[{args.engine}] engine={eng.name} model={model}", flush=True)

    native_src = open(os.path.join(_THIS, "seeds", "native.py")).read()
    native_state = {"src": native_src, "aep": base_mean, "pct": 0.0, "feas": True,
                    "n_feas": len(args.seeds), "max_bnd": 0.0, "min_dist": None,
                    "min_spacing": None, "viol": 0.0, "iter": 0}
    best = None                        # best STRICTLY-feasible schedule seen
    pool = []                          # top-K feasible states -> diverse parents
    traj = []
    start_it = 1
    ckpt_path = os.path.join(OUT, "summary.json")

    def _iter_src(n):
        p = os.path.join(OUT, f"iter_{n:03d}.py")
        return open(p).read() if os.path.exists(p) else None

    def _mk(it_no, pct, aep):
        return {"src": _iter_src(it_no), "pct": pct, "aep": aep, "iter": it_no,
                "feas": True, "n_feas": len(args.seeds), "max_bnd": 0.0,
                "min_dist": None, "min_spacing": None, "viol": 0.0}

    if args.resume and os.path.exists(ckpt_path):
        prev = json.load(open(ckpt_path))
        traj = prev.get("trajectory", [])
        start_it = len(traj) + 1
        feas = sorted([t for t in traj if t.get("feasible") and _iter_src(t["iter"])],
                      key=lambda t: t["pct"], reverse=True)
        for t in feas[:args.pool_k]:
            pool.append(_mk(t["iter"], t["pct"], t.get("aep")))
        # restore the recorded best even if it's the iter-0 anchor (absent from traj)
        bf = prev.get("best_feasible")
        if bf and _iter_src(bf["iter"]) is not None and \
                not any(p["iter"] == bf["iter"] for p in pool):
            pool.append(_mk(bf["iter"], bf["pct"], bf.get("aep")))
        pool = sorted([p for p in pool if p["src"] is not None],
                      key=lambda p: p["pct"], reverse=True)[:args.pool_k]
        best = dict(pool[0]) if pool else None
        print(f"[{args.engine}] RESUME: {len(traj)} attempts done, "
              f"best={('%+.4f%%' % best['pct']) if best else 'none'}", flush=True)
    elif args.seed_from:               # CONTINUE from a prior winner (evaluate it)
        seed_src = open(args.seed_from).read()
        s0 = _score(seed_src, args.seeds, args.steps)
        s0["pct"] = 100.0 * (s0["aep"] - base_mean) / base_mean
        s0["src"], s0["iter"] = seed_src, 0
        with open(os.path.join(OUT, "iter_000.py"), "w") as f:   # persist anchor for resume
            f.write(seed_src)
        if s0["feas"]:
            pool = [s0]; best = dict(s0)
        print(f"[{args.engine}] seeded from {args.seed_from}: "
              f"{s0['pct']:+.4f}% feasible={s0['feas']}", flush=True)

    def _fb(c):   # firewall-safe per-cell feedback dict shown to the mutator
        d = {"score_pct": round(c["pct"], 4), "feasible": c["feas"],
             "seeds_feasible": f"{c['n_feas']}/{len(args.seeds)}"}
        if not c["feas"]:
            d["boundary_penalty"] = round(c["max_bnd"], 6)
            if c["min_dist"] is not None:
                d["min_dist_m"] = round(c["min_dist"], 2)
        return {CELL: d}

    def _reflection():
        # ReEvo-style verbal-gradient: show the mutator the scoreboard it otherwise
        # never sees — the best-so-far (code + score) and the recent failed deltas —
        # so it stops orbiting the plateau and tries something structural.
        if not (args.reflect and best):
            return ""
        recent = [t for t in traj if "pct" in t][-8:]
        hist = ", ".join(f"{t['pct']:+.4f}%{'' if t.get('feasible') else '(infeas)'}"
                         for t in recent) or "(none yet)"
        return (
            f"SEARCH STATE — you MUST BEAT the current best.\n"
            f"Best schedule so far: {best['pct']:+.4f}% (feasible). Its code:\n"
            f"```python\n{best['src']}\n```\n"
            f"Scores of the most recent attempts (oldest→newest): {hist} — every one is "
            f"AT OR BELOW the best. Small peak/ramp/plateau-constant tweaks are NOT "
            f"breaking through.\n"
            f"So do NOT submit another minor variation of the best. Make a STRUCTURALLY "
            f"DIFFERENT schedule while keeping strict feasibility (a strong terminal alpha "
            f"restoration). Directions NOT yet tried: cyclic alpha with SGDR warm restarts; "
            f"a DECOUPLED penalty (alpha NOT tied to 1/lr — e.g. a logistic ramp-then-"
            f"plateau); a multi-cycle cosine lr; an ADMM-style constant moderate penalty; "
            f"mid-run feasibility-restoration bursts.\n\n")

    def _last_note(c):
        if c["feas"]:
            return (f"CURRENT PARENT: FEASIBLE at {c['pct']:+.4f}% AEP vs baseline "
                    f"({c['n_feas']}/{len(args.seeds)} seeds). Keep it STRICTLY feasible "
                    f"and push AEP higher (e.g. a slightly higher/longer lr peak early), "
                    f"but PRESERVE the terminal feasibility restoration.\n\n")
        md = f"{c['min_dist']:.1f}" if c['min_dist'] is not None else "?"
        ms = f"{c['min_spacing']:.0f}" if c['min_spacing'] else "?"
        return (f"CURRENT PARENT: INFEASIBLE — boundary_penalty={c['max_bnd']:.3e} "
                f"(must be ~0), min_dist={md} m vs required {ms} m, "
                f"{c['n_feas']}/{len(args.seeds)} seeds feasible. The layout drifted out "
                f"of bounds / turbines too close. FIX FEASIBILITY FIRST: start the terminal "
                f"alpha restoration EARLIER and STRONGER (raise terminal alpha well before "
                f"the final 8%), and keep lr high enough DURING the restoration so drifted "
                f"turbines can move back inside — a spike that fires only after lr has "
                f"collapsed to the floor cannot pull them back. Only then improve AEP.\n\n")

    pa = _prior_art() if args.prior_art else ""
    print(f"[{args.engine}] prior-art context injected: {bool(pa)} "
          f"({len(pa)} chars from PRIOR_ART.md §10)", flush=True)
    guidance = (
        "PRIOR-ART DESIGN GUIDANCE (frozen literature survey — apply where it helps).\n"
        "Especially relevant to FIXING boundary/spacing violations: the bounded/logistic "
        "alpha plateau, the DELAYED alpha ramp, and a TERMINAL FEASIBILITY SPIKE (drive "
        "alpha up + lr down in the final ~5-10% of steps).\n\n" + pa + "\n\n"
        if pa else "")

    def _ckpt():
        json.dump({"cell": CELL, "seeds": args.seeds, "baseline_mean": base_mean,
                   "prior_art": bool(pa), "engine": args.engine, "trajectory": traj,
                   "best_feasible": ({k: best[k] for k in ("pct", "aep", "iter")}
                                     if best else None),
                   "pool": [{"pct": p["pct"], "iter": p["iter"]} for p in pool]},
                  open(ckpt_path, "w"), indent=2)

    for it in range(start_it, args.iters + 1):
        rng = random.Random(f"parent:{it}")
        if it % args.restart_every == 0:        # explore a fresh basin
            parent = native_state
        elif pool and rng.random() < 0.35:      # diversify among the good ones
            parent = rng.choice(pool)
        else:                                   # exploit the best so far
            parent = best or native_state
        W.materialize(scope, parent_source=parent["src"], feedback=_fb(parent), fw2_root=_THIS)
        ctx = EvoContext(parent_source=W.sanitize(parent["src"]),
                         parent_id=f"it{parent['iter']}", generation=it, island=0,
                         per_cell_fitness=_fb(parent),
                         notes=guidance + _reflection() + _last_note(parent) + "You are "
                         "editing the schedule shown above (the parent). Return an improved "
                         "module defining only schedule_fn.")
        t0 = time.time()
        res = eng.mutate(ctx)
        child = res.source
        if not child or not _valid(child):
            print(f"[{args.engine}] iter {it}: no valid schedule ({res.log.error[:50]}) "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            traj.append({"iter": it, "status": "invalid", "parent": parent["iter"]})
            _ckpt(); continue
        with open(os.path.join(OUT, f"iter_{it:03d}.py"), "w") as f:
            f.write(child)
        sc = _score(child, args.seeds, args.steps)
        sc["pct"] = 100.0 * (sc["aep"] - base_mean) / base_mean
        sc["src"], sc["iter"] = child, it
        tag = "FEAS" if sc["feas"] else f"infeas(v={sc['viol']:.1e})"
        bpct = f"{best['pct']:+.4f}" if best else "+0.0000"
        print(f"[{args.engine}] iter {it}: {sc['pct']:+.4f}% {tag}  [best {bpct}%] "
              f"par=it{parent['iter']} [{time.time()-t0:.0f}s]", flush=True)
        traj.append({"iter": it, "aep": round(sc["aep"], 4), "pct": round(sc["pct"], 4),
                     "feasible": sc["feas"], "viol": round(sc["viol"], 6),
                     "parent": parent["iter"]})
        if sc["feas"]:
            pool = sorted(pool + [sc], key=lambda p: p["pct"], reverse=True)[:args.pool_k]
            if best is None or sc["pct"] > best["pct"]:
                best = sc
        _ckpt()

    print("\n=== SEARCH DONE ===", flush=True)
    nf = sum(1 for t in traj if t.get("feasible"))
    print(f"  attempts={len(traj)}  feasible={nf}  "
          f"best={('%+.4f%%' % best['pct']) if best else 'none'}", flush=True)
    if best is not None:
        bpath = os.path.join(OUT, f"iter_{best['iter']:03d}.py")
        print(f"  best: iter {best['iter']}  {best['pct']:+.4f}%  AEP {best['aep']:.4f}  -> {bpath}")
    _ckpt()


if __name__ == "__main__":
    main()
