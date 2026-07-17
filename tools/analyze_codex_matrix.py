"""Analyze the codex full-opt matrix eval (Task P) vs per-cell baselines and
the schedule-only dual-bump matrix.

Answers: does the richer optimize() interface generalize across the 64-cell
matrix, or only shatter the DEI training ceiling? Where do the 43 errors fall?
"""
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

codex = json.load(open(os.path.join(ROOT, "results/matrix/codex_fullopt_matrix.json")))
baselines = json.load(open(os.path.join(ROOT, "results/matrix/baselines_matrix.json")))

# schedule-only dual-bump matrix (if present) for comparison
sched_path = os.path.join(ROOT, "results/matrix/schedules_matrix.json")
sched = json.load(open(sched_path)) if os.path.exists(sched_path) else {}


def cellkey(farm, n, rose):
    return f"{farm}_n{n}_rose{rose}"


# Baseline lookup: baselines_matrix.json keyed how? inspect
bl = {}
for k, v in baselines.items():
    # try to normalize
    if isinstance(v, dict):
        aep = v.get("best_aep") or v.get("aep_gwh") or v.get("aep")
    else:
        aep = v
    bl[k] = aep

# Best codex champion per cell (max AEP among the 3, feasible only)
by_cell = defaultdict(list)
errors = []
for key, e in codex.items():
    if "aep_gwh" in e and e.get("feasible"):
        by_cell[cellkey(e["farm"], e["n"], e["rose"])].append((e["label"], e["aep_gwh"]))
    elif "error" in e:
        errors.append((e["farm"], e["n"], e["rose"], e["label"], e.get("error", "")[:40]))

print("=" * 78)
print("CODEX FULL-OPT MATRIX vs BASELINE")
print("=" * 78)

# Per-N, per-rose summary: how often does best codex champion beat baseline?
wins = defaultdict(lambda: [0, 0])  # (n): [beats, total]
rows = []
for key, champs in sorted(by_cell.items()):
    best_label, best_aep = max(champs, key=lambda x: x[1])
    b = bl.get(key)
    delta = (best_aep - b) if b else None
    rows.append((key, best_aep, b, delta, len(champs)))

# tabulate by N
print(f"\n{'cell':32} {'codex_best':>10} {'baseline':>10} {'delta':>9} {'nfeas/3':>7}")
print("-" * 78)
for key, best_aep, b, delta, nf in rows:
    bstr = f"{b:.1f}" if b else "  n/a"
    dstr = f"{delta:+.1f}" if delta is not None else "   n/a"
    flag = ""
    if delta is not None:
        flag = "  WIN" if delta > 0 else "  lose"
    print(f"{key:32} {best_aep:10.1f} {bstr:>10} {dstr:>9} {nf:>5}/3{flag}")

# Aggregate win-rate where baseline exists
have_b = [r for r in rows if r[3] is not None]
n_win = sum(1 for r in have_b if r[3] > 0)
print(f"\nCells with baseline: {len(have_b)}")
print(f"Codex-best beats baseline: {n_win}/{len(have_b)} "
      f"({100*n_win/len(have_b):.0f}%)")
mean_delta = sum(r[3] for r in have_b) / len(have_b)
print(f"Mean delta vs baseline: {mean_delta:+.1f} GWh")

# by rose
by_rose = defaultdict(lambda: [0, 0, 0.0])
for key, best_aep, b, delta, nf in rows:
    if delta is None:
        continue
    rose = key.split("rose")[1]
    by_rose[rose][0] += 1 if delta > 0 else 0
    by_rose[rose][1] += 1
    by_rose[rose][2] += delta
print("\nBy rose (win-rate, mean-delta):")
for rose, (w, t, sd) in sorted(by_rose.items()):
    print(f"  {rose:10} {w}/{t} win, mean {sd/t:+.1f} GWh")

# Errors
print(f"\n{'='*78}\n{len(errors)} ERRORS (champion × cell)\n{'='*78}")
err_by_n = defaultdict(int)
for farm, n, rose, label, msg in errors:
    err_by_n[n] += 1
for n in sorted(err_by_n):
    print(f"  N={n}: {err_by_n[n]} errors")
# sample error messages
print("sample errors:")
for farm, n, rose, label, msg in errors[:6]:
    print(f"  {farm}_n{n}_rose{rose} [{label.split('(')[0].strip()}]: {msg}")

# compare vs schedule dual-bump if present
if sched:
    print(f"\n{'='*78}\nCODEX FULL-OPT vs SCHEDULE DUAL-BUMP (per cell)\n{'='*78}")
    claude_sched = {k: v for k, v in sched.items() if "iter 192" in k and "Claude" in k}
    both = 0; codex_higher = 0
    for skey, sv in claude_sched.items():
        if "aep_gwh" not in sv or not sv.get("feasible"):
            continue
        ck = cellkey(sv["farm"], sv["n"], sv["rose"])
        if ck in by_cell:
            codex_best = max(by_cell[ck], key=lambda x: x[1])[1]
            both += 1
            if codex_best > sv["aep_gwh"]:
                codex_higher += 1
    if both:
        print(f"Cells both feasible: {both}")
        print(f"Codex full-opt > Claude dual-bump: {codex_higher}/{both} "
              f"({100*codex_higher/both:.0f}%)")
