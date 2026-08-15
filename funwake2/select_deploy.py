"""Deployment-selection metric (FunWake-1 style): choose which discovered schedule
to deploy by its HELD-OUT generalization, feasibility-gated — not by its training
score. The searcher never sees the held-out ROWP AEP, so ROWP@10 is the honest
generalization signal (mirrors FunWake-1, where the LLM optimizer was selected on
the held-out IEA-740-10 ROWP case, not on the DEI training farm).

Per candidate (read from its validation JSON produced by run_validation.py):
  GATE      : feasible on ALL seeds of the held-out test (rowp_n74); else DISQUALIFIED.
  PRIMARY   : rowp_n74 delta_pct vs native c*D  (held-out generalization — the score).
  robustness: farm-balanced mean delta_pct over ALL validation cells (context),
              worst-cell delta (downside), and #cells feasible.
Selected = the feasibility-gated argmax of PRIMARY.

Usage:  pixi run python funwake2/select_deploy.py
"""
import glob
import json
import os
import statistics

_THIS = os.path.dirname(os.path.abspath(__file__))
VDIR = os.path.join(_THIS, "state", "validation")

# friendly labels for known candidate validation files
LABELS = {
    "iter109.json": "Claude it109",
    "antigravity488.json": "Antigravity it488",
    "codex021.json": "Codex it21",
    "iter04.json": "Codex it04 (old)",
    "port190.json": "Portfolio it190",
}


def _feas(s):
    """'10/10' -> (10, 10)."""
    a, b = str(s).split("/")
    return int(a), int(b)


def _held_out_key(cells):
    for k in cells:
        if k.startswith("rowp_"):
            return k
    return None


def _gated(r):
    """Per-cell fitness contribution: real Δ% only if fully feasible, else a penalty.
    Mirrors run_portfolio_explore._eff_score — an infeasible layout is non-deployable,
    so its (often inflated) AEP must NOT be rewarded (adversarial review 1.1)."""
    a, b = _feas(r["cand_feasible"])
    return r["delta_pct"] if a == b else min(r["delta_pct"], -1.0)


def _paired_t(rows):
    """Paired-difference t-stat (cand-native AEP) over seeds — is the win real? (1.3)"""
    if not rows or "cand_aep" not in rows[0] or "native_aep" not in rows[0]:
        return None
    diffs = [r["cand_aep"] - r["native_aep"] for r in rows]
    n = len(diffs)
    if n < 2:
        return None
    m = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / (n ** 0.5)
    return {"mean_diff": m, "se": se, "t": (m / se if se > 0 else float("inf")), "n": n}


def summarize(path):
    d = json.load(open(path))
    cells = d["cells"]
    hk = _held_out_key(cells)
    if hk is None:
        return None
    ho = cells[hk]
    if "cand_feasible" not in ho or "delta_pct" not in ho:
        return None                            # not a run_validation-format file (skip)
    hf, ht = _feas(ho["cand_feasible"])
    deltas_raw = {k: r["delta_pct"] for k, r in cells.items()}
    deltas_gated = {k: _gated(r) for k, r in cells.items()}
    feas_cells = sum(1 for r in cells.values() if _feas(r["cand_feasible"])[0] == _feas(r["cand_feasible"])[1])
    worst = min(deltas_gated.values())
    worst_cell = min(deltas_gated, key=deltas_gated.get)
    tt = _paired_t(ho.get("rows"))
    return {
        "held_out": hk,
        "primary": ho["delta_pct"],            # PRIMARY score (held-out ROWP Δ%)
        "held_out_feasible": (hf == ht),       # feasibility GATE on the held-out farm
        "held_out_feas_str": ho["cand_feasible"],
        "ho_t": (tt["t"] if tt else None),     # paired-t of the held-out win (>~2 = real)
        "ho_significant": bool(tt and tt["t"] > 2.0),
        "fb_mean": statistics.fmean(deltas_gated.values()),      # FEASIBILITY-GATED mean (honest)
        "fb_mean_raw": statistics.fmean(deltas_raw.values()),    # ungated (gamed) mean — shown to expose the artifact
        "worst": worst, "worst_cell": worst_cell,
        "feas_cells": feas_cells, "n_cells": len(cells),
        "all_feasible": (feas_cells == len(cells)),
    }


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(VDIR, "*.json"))):
        name = os.path.basename(path)
        s = summarize(path)
        if s is None:
            continue
        s["label"] = LABELS.get(name, name)
        rows.append(s)

    # PRIMARY = held-out ROWP Δ% (the one out-of-portfolio farm), feasibility-gated and
    # significance-checked. fb_mean is shown FEASIBILITY-GATED (honest) alongside its
    # ungated value (which rewards infeasible non-deployable layouts — the gamed metric).
    eligible = [r for r in rows if r["held_out_feasible"]]
    eligible.sort(key=lambda r: r["primary"], reverse=True)
    dq = [r for r in rows if not r["held_out_feasible"]]

    print("=" * 96)
    print("FunWake deployment selection — PRIMARY = held-out ROWP Δ% (feasibility-gated + significance)")
    print("=" * 96)
    hdr = (f"{'candidate':20s} {'ROWP Δ%':>9s} {'t':>6s} {'sig':>4s} | "
           f"{'fb(gated)':>10s} {'fb(ungated)':>11s} {'worst':>8s} {'feas':>6s}")
    print(hdr)
    print("-" * 96)
    for r in eligible + dq:
        gate = "" if r["held_out_feasible"] else " DQ"
        t = f"{r['ho_t']:+.1f}" if r["ho_t"] is not None else "  n/a"
        sig = "yes" if r["ho_significant"] else "no"
        print(f"{r['label']:20s} {r['primary']:>+8.4f}%{gate:>0s} {t:>6s} {sig:>4s} | "
              f"{r['fb_mean']:>+9.4f}% {r['fb_mean_raw']:>+10.4f}% {r['worst']:>+7.4f}% "
              f"{r['feas_cells']}/{r['n_cells']:>d}")
    print("-" * 96)
    # deploy pick: best held-out ROWP among gate-passers whose held-out win is SIGNIFICANT
    sig_eligible = [r for r in eligible if r["ho_significant"] and r["primary"] > 0]
    if sig_eligible:
        w = sig_eligible[0]
        print(f"\n>>> DEPLOY (held-out generalization): {w['label']}  "
              f"(ROWP {w['primary']:+.4f}%, t={w['ho_t']:+.1f} significant, "
              f"feasible-cells {w['feas_cells']}/{w['n_cells']})")
        print(f"    rationale: the only clean out-of-portfolio held-out farm, feasibility-gated,")
        print(f"    with a paired win that excludes zero. Ranking by fb_mean is UNSAFE — it is")
        print(f"    in-sample (5/6 cells are training farms) and its ungated form rewards")
        print(f"    infeasible non-deployable layouts (see fb(gated) vs fb(ungated)).")
    else:
        print("\n>>> DEPLOY (held-out generalization): NONE — no gate-passing candidate has a")
        print("    statistically significant (t>2) positive held-out ROWP win. Deploy native c*D.")
    # honest fb_mean winner (gated), for contrast
    ef = sorted([r for r in rows if r["all_feasible"]], key=lambda r: r["fb_mean"], reverse=True)
    if ef:
        b = ef[0]
        print(f"\n    (robustness view — best FEASIBILITY-GATED fb_mean among ALL-feasible candidates: "
              f"{b['label']} {b['fb_mean']:+.4f}%)")

    # A single global champion ignores that farms differ (N, geometry, wind rose,
    # turbine type). The honest deployment decision is a SELECTION FUNCTION over farm
    # characteristics: for each farm type, deploy the best validated candidate, and
    # fall back to native c*D when none beats it. Two other metrics also shown, to
    # make explicit that the "winner" depends on what you optimise for.
    print("\n" + "=" * 82)
    print("METRIC SENSITIVITY — the pick depends on the objective")
    print("=" * 82)
    by_robust = sorted(rows, key=lambda r: r["fb_mean"], reverse=True)
    by_downside = sorted(rows, key=lambda r: r["worst"], reverse=True)
    print(f"  held-out ROWP (generalization) : {eligible[0]['label'] if eligible else '—'}")
    print(f"  farm-balanced mean (robustness): {by_robust[0]['label']}  ({by_robust[0]['fb_mean']:+.4f}%)")
    print(f"  best worst-case (min downside) : {by_downside[0]['label']}  ({by_downside[0]['worst']:+.4f}%)")

    print("\n" + "=" * 82)
    print("PER-FARM ROUTING — deploy the winner for each farm type (native fallback)")
    print("=" * 82)
    # union of cells across candidates; route each to its argmax (native if all <=0)
    all_cells = sorted({c for r in _RAW.values() for c in r})
    print(f"  {'farm cell':24s} {'winner':20s} {'Δ%':>9s}   deploy")
    for c in all_cells:
        vals = {lbl: cells[c]["delta_pct"] for lbl, cells in _RAW.items() if c in cells}
        if not vals:
            continue
        win = max(vals, key=vals.get)
        wv = vals[win]
        print(f"  {c:24s} {win:20s} {wv:>+8.4f}%   -> {win if wv > 0 else 'native c*D'}")


_RAW = {}


if __name__ == "__main__":
    # cache raw cell dicts (labelled) for the routing section
    for _p in sorted(glob.glob(os.path.join(VDIR, "*.json"))):
        _n = os.path.basename(_p)
        if _held_out_key(json.load(open(_p))["cells"]):
            _RAW[LABELS.get(_n, _n)] = json.load(open(_p))["cells"]
    main()


