"""Portfolio (multi-farm) exploration — the generalization experiment.

Instead of scoring candidates on ONE farm (which yields single-farm SPECIALISTS,
see the validation of the single-cell runs), score each candidate on a DIVERSE
FARM PORTFOLIO spanning turbine count, wind rose, and geometry, with FARM-BALANCED
fitness. This pressures the search to discover a schedule that USES its per-farm
inputs (D, n_turbines, min_spacing, alpha0) to generalize, rather than baking in
one farm's characteristics.

Fitness       = mean over farms of per-farm score_c (%-over-native-c*D).
Feasibility   = must be feasible on ALL farms (hard gate); else ranked by violation.
Firewall      = portfolio is TRAINING cells only; ROWP (holdout/test) never appears
                here, and only per-farm %-over-baseline + feasibility booleans reach
                the mutator (never a raw holdout AEP).

  pixi run python funwake2/run_portfolio_explore.py --engine claude --tag port_claude \
      --iters 200 --seeds 0 1 2 --steps 8000

Budget is measured in REAL scored attempts (rate-limit safe, same as the single-cell
driver): a no-code mutation backs off and does not consume the budget.
"""
import argparse
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for _p in (_ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluator as E                                            # noqa: E402
from funwake2.controller import workspace as W                  # noqa: E402
from funwake2.controller.engines.base import EvoContext          # noqa: E402
from funwake2.controller.engines.codex_cli import CodexCLIEngine  # noqa: E402
from funwake2.controller.engines.claude_cli import ClaudeCLIEngine  # noqa: E402
from funwake2.controller.engines.gemini_cli import GeminiCLIEngine  # noqa: E402
from funwake2.controller.engines.antigravity_cli import AntigravityCLIEngine  # noqa: E402

_ENGINES = {"codex": (CodexCLIEngine, "gpt-5.5"),
            "claude": (ClaudeCLIEngine, "claude-opus-4-8"),
            "gemini": (GeminiCLIEngine, "gemini-2.5-flash"),
            "antigravity": (AntigravityCLIEngine, "Gemini 3.1 Pro (High)")}

# Diverse training portfolio (ROWP held out). Spans N=10..80, wind=DEI-rose/uniform/
# omnidir, geometry=DEI-polygon/Parque-multizone. All have cached native baselines.
DEFAULT_CELLS = ["dei_n50", "dei_n80_omnidir", "dei_n50_uniform",
                 "parque_n20", "parque_n10_omnidir"]
OUT = None


def _native(cells, seeds):
    b = json.load(open(os.path.join(_THIS, "controller", "baselines_g2.json")))["cells"]
    return {c: {s: b[c]["seeds"][str(s)] for s in seeds} for c in cells}


def _valid(src):
    import ast
    try:
        t = ast.parse(src)
    except SyntaxError:
        return False
    return any(getattr(n, "name", "") == "schedule_fn" for n in ast.walk(t))


def _score(src, cells, seeds, steps, native, timeout_s=450):
    """Farm-balanced multi-cell score. Returns fb_mean (%), per-cell breakdown,
    all-feasible gate, and an infeasibility magnitude for ranking infeasibles."""
    path = os.path.join(OUT, "_cand.py")
    with open(path, "w") as f:
        f.write(src)
    fn = E.load_schedule(path)
    have_alarm = hasattr(signal, "SIGALRM")
    if have_alarm:
        def _to(signum, frame):
            raise TimeoutError(f"eval > {timeout_s}s ({len(cells)}x{len(seeds)})")
        _old = signal.signal(signal.SIGALRM, _to)
        signal.alarm(int(timeout_s))
    try:
        per = {}
        for c in cells:
            recs = [E.evaluate(c, fn, seed=s, total_steps=steps, gamma_min=0.01) for s in seeds]
            aeps = [r["aep_gwh"] for r in recs]
            feas = all(r["feasible"] for r in recs)
            cand_mean = statistics.fmean(aeps)
            nat_mean = statistics.fmean(native[c][s] for s in seeds)
            ms = float(recs[0].get("min_spacing") or 0.0)
            max_bnd = max(float(r.get("boundary_penalty", 0.0) or 0.0) for r in recs)
            min_dist = min(float(r.get("min_dist_m", 1e9) or 1e9) for r in recs)
            short = max(0.0, (ms - min_dist) / ms) if ms else 0.0
            per[c] = {"score_c": 100.0 * (cand_mean - nat_mean) / nat_mean,
                      "feas": feas, "n_feas": sum(1 for r in recs if r["feasible"]),
                      "max_bnd": max_bnd, "min_dist": min_dist, "min_spacing": ms,
                      "viol_c": 0.0 if feas else (max_bnd + short)}
    finally:
        if have_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, _old)
    all_feas = all(per[c]["feas"] for c in cells)
    fb_mean = statistics.fmean(per[c]["score_c"] for c in cells)      # farm-balanced
    worst = min(per[c]["score_c"] for c in cells)
    viol = 0.0 if all_feas else sum(per[c]["viol_c"] for c in cells)
    return {"pct": fb_mean, "worst": worst, "feas": all_feas, "viol": viol,
            "n_feas_cells": sum(1 for c in cells if per[c]["feas"]), "per": per}


# ---------------------------------------------------------------------------
# gbar eval backend: dispatch each candidate to a persistent worker on a DTU
# hpc compute node via a shared-filesystem queue. Same score aggregation, but the
# 5x3 evals run in parallel there (~one eval's wall-time) instead of sequentially.
# LLM mutation stays local (firewall). Baselines are gbar-native (same-platform).
# ---------------------------------------------------------------------------
# plain BatchMode SSH (ControlMaster multiplexing tripped on the DTU login banner);
# each candidate uses ONE blocking SSH that polls server-side for its result.
_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
_QDIR = "~/funwake/gbar_queue"
_gbar_ctr = [0]


def _aggregate(raw_cells, cells, seeds, native):
    """Build the same score dict as _score from raw per-(cell,seed) AEP/feasibility."""
    per = {}
    for c in cells:
        cs = raw_cells[c]
        aeps = [cs[str(s)]["aep"] for s in seeds]
        feasl = [bool(cs[str(s)]["feasible"]) for s in seeds]
        feas = all(feasl)
        cand_mean = statistics.fmean(aeps)
        nat_mean = statistics.fmean(native[c][s] for s in seeds)
        ms = float(cs[str(seeds[0])].get("min_spacing") or 0.0)
        max_bnd = max(float(cs[str(s)].get("boundary_penalty", 0.0) or 0.0) for s in seeds)
        min_dist = min(float(cs[str(s)].get("min_dist_m", 1e9) or 1e9) for s in seeds)
        short = max(0.0, (ms - min_dist) / ms) if ms else 0.0
        per[c] = {"score_c": 100.0 * (cand_mean - nat_mean) / nat_mean, "feas": feas,
                  "n_feas": sum(1 for x in feasl if x), "max_bnd": max_bnd,
                  "min_dist": min_dist, "min_spacing": ms,
                  "viol_c": 0.0 if feas else (max_bnd + short)}
    all_feas = all(per[c]["feas"] for c in cells)
    fb_mean = statistics.fmean(per[c]["score_c"] for c in cells)
    worst = min(per[c]["score_c"] for c in cells)
    viol = 0.0 if all_feas else sum(per[c]["viol_c"] for c in cells)
    return {"pct": fb_mean, "worst": worst, "feas": all_feas, "viol": viol,
            "n_feas_cells": sum(1 for c in cells if per[c]["feas"]), "per": per}


def _fetch_gbar_baselines(cells, seeds, host="gbar", wait_s=2400):
    """Pull the gbar-native baselines (worker bootstraps them on first start)."""
    t0 = time.time()
    while time.time() - t0 < wait_s:
        p = subprocess.run(_SSH + [host, "cat ~/funwake/gbar_native_baselines.json 2>/dev/null"],
                           capture_output=True, text=True, timeout=60)
        if p.stdout.strip():
            try:
                d = json.loads(p.stdout)
            except json.JSONDecodeError:
                time.sleep(10); continue
            return {c: {s: d["cells"][c][str(s)]["aep"] for s in seeds} for c in cells}
        print("[gbar] waiting for native baselines (worker bootstrapping)...", flush=True)
        time.sleep(20)
    raise RuntimeError("gbar native baselines not ready (worker not started?)")


def _score_gbar(src, cells, seeds, steps, native, host="gbar", jobs=15, timeout_s=900):
    """Dispatch one candidate to the gbar worker; block until scored; aggregate."""
    _gbar_ctr[0] += 1
    rid = f"{os.getpid()}_{_gbar_ctr[0]}"
    req_local = os.path.join(OUT, f"_req_{rid}.py")
    with open(req_local, "w") as f:
        f.write(src)
    # ship atomically: write to a dotfile, then rename into req_<id>.py (worker globs req_*)
    with open(req_local, "rb") as f:
        subprocess.run(_SSH + [host, f"cat > {_QDIR}/.tmp_{rid}.py && mv {_QDIR}/.tmp_{rid}.py "
                               f"{_QDIR}/req_{rid}.py"], stdin=f, check=True, timeout=90)
    os.remove(req_local)
    # ONE blocking SSH: poll server-side for the resp, print it when ready
    poll = (f"for i in $(seq 1 {max(1, timeout_s // 3)}); do "
            f"if [ -f {_QDIR}/resp_{rid}.json ]; then cat {_QDIR}/resp_{rid}.json; exit 0; fi; "
            f"sleep 3; done; exit 7")
    raw = None
    try:
        p = subprocess.run(_SSH + [host, poll], capture_output=True, text=True,
                           timeout=timeout_s + 60)
        if p.returncode == 0 and p.stdout.strip():
            raw = json.loads(p.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        raw = None
    subprocess.run(_SSH + [host, f"rm -f {_QDIR}/resp_{rid}.json {_QDIR}/req_{rid}.py "
                           f"{_QDIR}/proc_{rid}.py"], timeout=60)
    if raw is None:
        raise TimeoutError(f"gbar eval timeout ({timeout_s}s)")
    return _aggregate(raw["cells"], cells, seeds, native)


def _fb(sc, cells, seeds):
    """Firewall-safe per-cell feedback (training cells only)."""
    out = {}
    for c in cells:
        cd = sc["per"][c]
        e = {"score_pct": round(cd["score_c"], 4), "feasible": cd["feas"],
             "seeds_feasible": f"{cd['n_feas']}/{len(seeds)}"}
        if not cd["feas"]:
            e["boundary_penalty"] = round(cd["max_bnd"], 6)
            if cd["min_dist"] is not None:
                e["min_dist_m"] = round(cd["min_dist"], 2)
        out[c] = e
    return out


def _prior_art():
    import re
    p = os.path.join(_THIS, "PRIOR_ART.md")
    if not os.path.exists(p):
        return ""
    m = re.search(r"## 10\. Synthesis.*?(?=\n## 11\.)", open(p).read(), re.S)
    return m.group(0).strip() if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--cells", nargs="+", default=DEFAULT_CELLS)
    ap.add_argument("--engine", choices=list(_ENGINES), default="claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed-from", dest="seed_from", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pool-k", dest="pool_k", type=int, default=6)
    ap.add_argument("--restart-every", dest="restart_every", type=int, default=12)
    ap.add_argument("--prior-art", dest="prior_art", action="store_true", default=True)
    ap.add_argument("--no-prior-art", dest="prior_art", action="store_false")
    ap.add_argument("--reflect", action="store_true", default=True)
    ap.add_argument("--no-reflect", dest="reflect", action="store_false")
    ap.add_argument("--eval-backend", dest="eval_backend", choices=["local", "gbar"],
                    default="local", help="score locally, or dispatch to the gbar worker")
    ap.add_argument("--gbar-host", dest="gbar_host", default="gbar")
    ap.add_argument("--gbar-jobs", dest="gbar_jobs", type=int, default=15)
    args = ap.parse_args()

    global OUT
    OUT = os.path.join(_THIS, "state", "explore_" + (args.tag or ("port_" + args.engine)))
    os.makedirs(OUT, exist_ok=True)
    cells, seeds = args.cells, args.seeds
    if args.eval_backend == "gbar":
        native = _fetch_gbar_baselines(cells, seeds, host=args.gbar_host)
        def SCORE(src, c, s, st, nat):
            return _score_gbar(src, c, s, st, nat, host=args.gbar_host, jobs=args.gbar_jobs)
    else:
        native = _native(cells, seeds)
        SCORE = _score
    EngCls, default_model = _ENGINES[args.engine]
    model = args.model or default_model
    scope = os.path.join(OUT, "scope")
    eng = EngCls(model=model, cwd=scope)
    eng.preflight()
    print(f"[{args.engine}] PORTFOLIO engine={eng.name} model={model} backend={args.eval_backend}", flush=True)
    print(f"[{args.engine}] farms={cells} seeds={seeds} (farm-balanced fitness)", flush=True)

    native_src = open(os.path.join(_THIS, "seeds", "native.py")).read()
    ns = {"src": native_src, "pct": 0.0, "worst": 0.0, "feas": True, "viol": 0.0,
          "n_feas_cells": len(cells), "iter": 0,
          "per": {c: {"score_c": 0.0, "feas": True, "n_feas": len(seeds),
                      "max_bnd": 0.0, "min_dist": None, "min_spacing": None,
                      "viol_c": 0.0} for c in cells}}
    best = None
    pool = []
    traj = []
    ckpt_path = os.path.join(OUT, "summary.json")

    def _iter_src(n):
        p = os.path.join(OUT, f"iter_{n:03d}.py")
        return open(p).read() if os.path.exists(p) else None

    def _mk(t):
        # trajectory stores per as a flat {cell: score_c}; rebuild the full per-cell
        # dict shape (score_c/feas/...) that _fb and _note expect (feasible parent).
        flat = t.get("per", {})
        per = {c: {"score_c": flat.get(c, 0.0), "feas": True, "n_feas": len(seeds),
                   "max_bnd": 0.0, "min_dist": None, "min_spacing": None, "viol_c": 0.0}
               for c in cells}
        return {"src": _iter_src(t["iter"]), "pct": t["pct"], "worst": t.get("worst"),
                "iter": t["iter"], "feas": True, "viol": 0.0,
                "n_feas_cells": len(cells), "per": per}

    if args.resume and os.path.exists(ckpt_path):
        prev = json.load(open(ckpt_path))
        traj = prev.get("trajectory", [])
        feas = sorted([t for t in traj if t.get("feasible") and _iter_src(t["iter"])],
                      key=lambda t: t["pct"], reverse=True)
        for t in feas[:args.pool_k]:
            pool.append(_mk(t))
        bf = prev.get("best_feasible")
        if bf and _iter_src(bf["iter"]) is not None and not any(p["iter"] == bf["iter"] for p in pool):
            pool.append(_mk(bf))
        pool = sorted([p for p in pool if p["src"]], key=lambda p: p["pct"], reverse=True)[:args.pool_k]
        best = dict(pool[0]) if pool else None
        print(f"[{args.engine}] RESUME: {len(traj)} attempts, "
              f"best={('%+.4f%%' % best['pct']) if best else 'none'}", flush=True)
    elif args.seed_from:
        seed_src = open(args.seed_from).read()
        s0 = SCORE(seed_src, cells, seeds, args.steps, native)
        s0.update({"src": seed_src, "iter": 0})
        with open(os.path.join(OUT, "iter_000.py"), "w") as f:
            f.write(seed_src)
        if s0["feas"]:
            pool = [s0]; best = dict(s0)
        print(f"[{args.engine}] seeded {args.seed_from}: fb={s0['pct']:+.4f}% "
              f"feas={s0['feas']} worst={s0['worst']:+.4f}%", flush=True)

    def _cellsummary(sc):
        return "  ".join(f"{c.split('_')[0][:4]}{('_'+c.split('_')[-1]) if 'uniform' in c or 'omnidir' in c else ''}"
                         f"={sc['per'][c]['score_c']:+.3f}%{'' if sc['per'][c]['feas'] else 'X'}"
                         for c in cells)

    def _reflection():
        if not (args.reflect and best):
            return ""
        recent = [t for t in traj if "pct" in t][-8:]
        hist = ", ".join(f"{t['pct']:+.3f}%" for t in recent) or "(none)"
        return (f"SEARCH STATE — beat the current best FARM-BALANCED score.\n"
                f"Best so far: farm-balanced {best['pct']:+.4f}% (worst farm {best.get('worst',0):+.4f}%), "
                f"feasible on all farms. Its code:\n```python\n{best['src']}\n```\n"
                f"Recent farm-balanced scores: {hist}.\n"
                f"KEY: this must work across farms of DIFFERENT turbine counts, wind roses, and "
                f"geometries — use the n_turbines / D / min_spacing / alpha0 inputs to ADAPT, do "
                f"not hardcode one farm's behavior. Improving one farm while regressing another "
                f"does NOT raise the farm-balanced mean.\n\n")

    def _note(parent):
        per = parent["per"]
        bad = [c for c in cells if not per[c]["feas"]]
        neg = sorted([c for c in cells if per[c]["feas"] and per[c]["score_c"] < 0],
                     key=lambda c: per[c]["score_c"])
        s = (f"CURRENT PARENT farm-balanced {parent['pct']:+.4f}% "
             f"({parent['n_feas_cells']}/{len(cells)} farms feasible).\n")
        if bad:
            s += (f"INFEASIBLE on: {', '.join(bad)} — fix feasibility FIRST (stronger/earlier "
                  f"terminal alpha restoration; keep lr high enough during restoration to pull "
                  f"turbines back in bounds).\n")
        if neg:
            negstr = ", ".join(f"{c}({per[c]['score_c']:+.3f}%)" for c in neg[:4])
            s += (f"Feasible but BELOW native on: {negstr} — these drag the mean down; adapt the "
                  f"schedule to them (they differ in N / wind rose / geometry from the farms you "
                  f"do well on).\n")
        return s + "\n"

    pa = _prior_art() if args.prior_art else ""
    guidance = ("PRIOR-ART DESIGN GUIDANCE (frozen survey — apply where it helps).\n" + pa + "\n\n"
                if pa else "")

    def _ckpt():
        json.dump({"cells": cells, "seeds": seeds, "engine": args.engine,
                   "farm_balanced": True, "trajectory": traj,
                   "best_feasible": ({"pct": best["pct"], "worst": best.get("worst"),
                                      "iter": best["iter"], "per_cell":
                                      {c: round(best["per"][c]["score_c"], 4) for c in cells}}
                                     if best else None),
                   "pool": [{"pct": p["pct"], "iter": p["iter"]} for p in pool]},
                  open(ckpt_path, "w"), indent=2)

    it = max((x.get("iter", 0) for x in traj), default=0) + 1
    consec_invalid = 0
    while len(traj) < args.iters:
        rng = random.Random(f"parent:{it}")
        if it % args.restart_every == 0:
            parent = ns
        elif pool and rng.random() < 0.35:
            parent = rng.choice(pool)
        else:
            parent = best or ns
        W.materialize(scope, parent_source=parent["src"], feedback=_fb(parent, cells, seeds), fw2_root=_THIS)
        ctx = EvoContext(parent_source=W.sanitize(parent["src"]),
                         parent_id=f"it{parent['iter']}", generation=it, island=0,
                         per_cell_fitness=_fb(parent, cells, seeds),
                         notes=guidance + _reflection() + _note(parent) + "You are editing "
                         "the schedule above (the parent). Return an improved module defining "
                         "only schedule_fn that raises the FARM-BALANCED score across all farms.")
        t0 = time.time()
        res = eng.mutate(ctx)
        child = res.source
        if not child or not _valid(child):
            consec_invalid += 1
            backoff = min(120, 5 * 2 ** min(consec_invalid - 1, 5))
            print(f"[{args.engine}] iter {it}: no valid schedule ({res.log.error[:40]}) "
                  f"invalid#{consec_invalid} backoff {backoff}s", flush=True)
            it += 1
            if consec_invalid >= 15:
                print(f"[{args.engine}] 15 consecutive failures — aborting; resume later.", flush=True)
                break
            time.sleep(backoff)
            continue
        consec_invalid = 0
        with open(os.path.join(OUT, f"iter_{it:03d}.py"), "w") as f:
            f.write(child)
        try:
            sc = SCORE(child, cells, seeds, args.steps, native)
        except Exception as e:
            print(f"[{args.engine}] iter {it}: EVAL ERROR ({type(e).__name__}: {str(e)[:60]})", flush=True)
            traj.append({"iter": it, "status": "eval-error", "parent": parent["iter"]})
            _ckpt(); it += 1; continue
        sc.update({"src": child, "iter": it})
        tag = "FEAS" if sc["feas"] else f"infeas({sc['n_feas_cells']}/{len(cells)})"
        bpct = f"{best['pct']:+.4f}" if best else "+0.0000"
        print(f"[{args.engine}] iter {it}: fb={sc['pct']:+.4f}% worst={sc['worst']:+.4f}% {tag} "
              f"[best {bpct}%] par=it{parent['iter']} [{time.time()-t0:.0f}s] | {_cellsummary(sc)}",
              flush=True)
        traj.append({"iter": it, "pct": round(sc["pct"], 4), "worst": round(sc["worst"], 4),
                     "feasible": sc["feas"], "viol": round(sc["viol"], 6),
                     "per": {c: round(sc["per"][c]["score_c"], 4) for c in cells},
                     "parent": parent["iter"]})
        if sc["feas"]:
            pool = sorted(pool + [sc], key=lambda p: p["pct"], reverse=True)[:args.pool_k]
            if best is None or sc["pct"] > best["pct"]:
                best = sc
        _ckpt(); it += 1

    print("\n=== PORTFOLIO SEARCH DONE ===", flush=True)
    nf = sum(1 for t in traj if t.get("feasible"))
    print(f"  attempts={len(traj)} all-farm-feasible={nf} "
          f"best={('%+.4f%%' % best['pct']) if best else 'none'}", flush=True)
    if best:
        print(f"  best: iter {best['iter']} farm-balanced {best['pct']:+.4f}% "
              f"(worst farm {best.get('worst',0):+.4f}%) -> {OUT}/iter_{best['iter']:03d}.py")
    _ckpt()


if __name__ == "__main__":
    main()
