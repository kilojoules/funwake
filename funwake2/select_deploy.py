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


def summarize(path):
    d = json.load(open(path))
    cells = d["cells"]
    hk = _held_out_key(cells)
    if hk is None:
        return None
    ho = cells[hk]
    hf, ht = _feas(ho["cand_feasible"])
    deltas = {k: r["delta_pct"] for k, r in cells.items()}
    feas_cells = sum(1 for r in cells.values() if _feas(r["cand_feasible"])[0] == _feas(r["cand_feasible"])[1])
    worst = min(deltas.values())
    worst_cell = min(deltas, key=deltas.get)
    return {
        "held_out": hk,
        "primary": ho["delta_pct"],            # PRIMARY score (held-out ROWP Δ%)
        "held_out_feasible": (hf == ht),       # GATE
        "held_out_feas_str": ho["cand_feasible"],
        "fb_mean": statistics.fmean(deltas.values()),   # farm-balanced mean (context)
        "worst": worst, "worst_cell": worst_cell,
        "feas_cells": feas_cells, "n_cells": len(cells),
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

    # eligible = passes the held-out feasibility gate; rank by PRIMARY desc
    eligible = [r for r in rows if r["held_out_feasible"]]
    eligible.sort(key=lambda r: r["primary"], reverse=True)
    dq = [r for r in rows if not r["held_out_feasible"]]

    print("=" * 82)
    print("FunWake deployment selection — PRIMARY = held-out ROWP Δ% (feasibility-gated)")
    print("=" * 82)
    hdr = f"{'candidate':20s} {'ROWP(test)':>11s} {'gate':>7s} {'fb-mean':>9s} {'worst-cell':>22s} {'feas':>6s}"
    print(hdr)
    print("-" * 82)
    for r in eligible + dq:
        gate = "PASS" if r["held_out_feasible"] else "DQ"
        print(f"{r['label']:20s} {r['primary']:>+10.4f}% {gate:>7s} {r['fb_mean']:>+8.4f}% "
              f"{r['worst']:>+8.4f}% {r['worst_cell'][:12]:>12s} {r['feas_cells']}/{r['n_cells']:>d}")
    print("-" * 82)
    if eligible:
        w = eligible[0]
        print(f"\n>>> GLOBAL DEPLOY (single champion): {w['label']}  (held-out ROWP "
              f"{w['primary']:+.4f}%, feasible {w['held_out_feas_str']})")
        print(f"    rationale: best held-out generalization among feasibility-gated candidates.")
        if w["fb_mean"] < 0:
            print(f"    caveat: farm-balanced mean over all cells is {w['fb_mean']:+.4f}% "
                  f"(worst {w['worst']:+.4f}% on {w['worst_cell']}) — a DEI/ROWP-family")
            print(f"    specialist, NOT a universal optimizer.")
    else:
        print("\n>>> GLOBAL DEPLOY: none passed the held-out feasibility gate.")

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


