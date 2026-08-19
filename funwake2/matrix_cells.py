"""Register the results/matrix DEI farms (dei_n{N}_rose{R}) as evaluator cells for
the large-portfolio scaling experiment (train/test across the full N x wind-rose
matrix). ADDITIVE ONLY: adds role='train_matrix' cells; never touches the frozen
stage_b set or the holdout/test cells. Each matrix problem file is self-contained
(boundary + N turbines + wind rose baked in), so rose=None / multizone=False.
"""
import glob
import os
import re


def register(E=None):
    if E is None:
        import evaluator as E
    root = E.ROOT
    added = 0
    for farm in ("dei", "rowp"):
        pat = os.path.join(root, f"results/matrix/problem_{farm}_n*_rose*.json")
        for f in sorted(glob.glob(pat)):
            m = re.search(rf"problem_{farm}_n(\d+)_rose([a-z]+)\.json$", os.path.basename(f))
            if not m:
                continue
            n, rose = int(m.group(1)), m.group(2)
            name = f"{farm}_n{n}_rose{rose}"
            if name in E.CELLS:
                continue
            E.CELLS[name] = {
                "problem": f"results/matrix/problem_{farm}_n{n}_rose{rose}.json",
                "rose": None, "n": n, "multizone": False,
                "role": "train_matrix", "stage_a": False, "stage_b": False}
            added += 1
    return added
