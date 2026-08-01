"""Controller-machinery unit tests (jax-free, deterministic).

Validates, without any real optimizer eval or LLM call:
  * archive binning — the 3 self-contained ancestors occupy >=3 distinct cells;
    the cyclic (iter118-family) ancestor lands in the peak_lr/D > 1.2 bin (R4).
  * cascade stage ORDER + fast-reject.
  * BLOCKING fitness patch — an INFEASIBLE reference still yields a %-score
    (scale constant); the candidate's own feasibility is the hard gate.
  * cost ceiling — 90% abort fires.
  * checkpoint -> kill -> resume reproduces the archive + lineage BIT-IDENTICALLY
    with content-addressed cache hits (no recompute).
  * firewall — stage-C returns no raw holdout AEP.
"""
import json
import os
import shutil
import sys
import tempfile

_THIS = os.path.dirname(os.path.abspath(__file__))
_CTRL = os.path.dirname(_THIS)
_FW2 = os.path.dirname(_CTRL)
_ROOT = os.path.dirname(_FW2)
for _p in (_ROOT, _FW2):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from funwake2.controller import config as C
from funwake2.controller.cascade import Cascade
from funwake2.controller.controller import Controller, GEN0_SEEDS, _import_fn, _SEED_DIR
from funwake2.controller.descriptors import compute_descriptors, bin_descriptors
from funwake2.controller.cost import CostTracker
from funwake2.controller.engines.mock import MockEngine
from funwake2.controller.run_dry import make_fake_eval


def test_archive_binning():
    coords, peak_bins = [], {}
    for cid, fname, anc in GEN0_SEEDS:
        fn = _import_fn(os.path.join(_SEED_DIR, fname))
        desc = compute_descriptors(fn)
        coord = bin_descriptors(desc)
        coords.append(coord)
        peak_bins[fname] = (desc["peak_lr_over_D"], coord[0])
    distinct = set(coords)
    assert len(distinct) >= 3, f"ancestors must occupy >=3 cells, got {peak_bins}"
    # cyclic (iter118-family) peak_lr/D > 1.2 -> bin index 3
    assert peak_bins["seed_cyclic.py"][1] == 3, peak_bins
    return {"distinct_cells": len(distinct), "peak_bins": peak_bins}


def test_cost_abort():
    ct = CostTracker(max_usd=1.0, max_tokens=10**9, abort_fraction=0.90)
    ct.add(0.5, 1000)
    assert not ct.should_abort()
    ct.add(0.45, 1000)          # cumulative 0.95 -> >= 0.90
    assert ct.should_abort(), "abort must fire at 90% of MAX_USD"
    return {"usd_frac": round(ct.usd_frac, 3), "reason": ct.reason()}


def _fake_baselines():
    cells = {}
    for cell, base in [("dei_n50", 5560.0), ("parque_n20", 231.0),
                       ("parque_n10_omnidir", 118.0), ("dei_n80_omnidir", 8600.0)]:
        cells[cell] = {"seeds": {str(s): base for s in range(5)},
                       "mean": base, "feas": "ref"}
    return {"cells": cells}


def test_cascade_order_and_fast_reject():
    calls = []
    base_fake = make_fake_eval()

    def logging_eval(cell, fn, seed=0, total_steps=8000, gamma_min=0.01):
        calls.append((cell, seed))
        return base_fake(cell, fn, seed, total_steps, gamma_min)

    casc = Cascade(_dry_cfg(tempfile.mkdtemp()), evaluate_fn=logging_eval,
                   baselines=_fake_baselines())
    src = open(os.path.join(_SEED_DIR, "seed_native.py")).read()
    a = casc.stage_a(src, ["dei_n50", "parque_n20"], [0, 1])
    n_after_a = len(calls)
    b = casc.stage_b(src, ["dei_n50", "parque_n20"], [0, 1])
    assert n_after_a == 4, "stage A = 2 cells x 2 seeds"
    assert a.passed and b.passed
    # fast-reject: a schedule forced infeasible on a stage-A cell fails A.
    # SEPARATE cache dir — the content-addressed key is (schedule,cell,seed,...),
    # so a shared cache would (correctly) return the other cascade's result.
    casc2 = Cascade(_dry_cfg(tempfile.mkdtemp()),
                    evaluate_fn=make_fake_eval(infeasible_cells=["dei_n50"]),
                    baselines=_fake_baselines())
    a2 = casc2.stage_a(src, ["dei_n50", "parque_n20"], [0, 1])
    assert not a2.passed, "fast-reject must fail an infeasible stage-A cell"
    return {"stageA_evals": n_after_a, "stageA_pass": a.passed,
            "stageB_pass": b.passed, "fast_reject_works": (not a2.passed)}


def test_candidate_infeasible_one_stage_b_cell_fails():
    """Item-3 hard gate: a candidate FEASIBLE (and positive-scoring) in every
    stage-B cell but ONE, where it is infeasible, must FAIL stage B — the global
    per-cell feasibility gate. Coherent only because all stage-B references are
    themselves all-feasible (guaranteed by the parque_n30_uniform -> n14 swap)."""
    cells = ["dei_n50", "parque_n20", "parque_n10_omnidir"]
    # baseline (all-feasible references) present for every cell
    base = _fake_baselines()

    # (a) candidate feasible everywhere -> stage B PASSES
    ok = Cascade(_dry_cfg(tempfile.mkdtemp()), evaluate_fn=make_fake_eval(),
                 baselines=base)
    src = open(os.path.join(_SEED_DIR, "seed_cosine.py")).read()
    b_ok = ok.stage_b(src, cells, [0, 1])
    assert b_ok.passed, b_ok.per_cell

    # (b) SAME candidate made infeasible in exactly ONE cell -> stage B FAILS
    bad = Cascade(_dry_cfg(tempfile.mkdtemp()),
                  evaluate_fn=make_fake_eval(infeasible_cells=["parque_n10_omnidir"]),
                  baselines=base)
    b_bad = bad.stage_b(src, cells, [0, 1])
    assert not b_bad.passed, "one infeasible stage-B cell must fail the hard gate"
    assert not b_bad.per_cell["parque_n10_omnidir"]["feasible"]
    assert b_bad.per_cell["parque_n10_omnidir"]["score"] == float("-inf")
    # the other cells are still individually feasible — it's the GLOBAL gate that
    # rejects, not those cells
    assert b_bad.per_cell["dei_n50"]["feasible"]
    assert b_bad.per_cell["parque_n20"]["feasible"]
    return {"pass_when_all_feasible": b_ok.passed,
            "fail_when_one_infeasible": (not b_bad.passed),
            "gate_note": b_bad.notes}


def test_feasibility_only_cell_excluded_from_aggregate():
    """Round-2 item 1: a FEASIBILITY-ONLY cell (parque_n14_uniform, saturated
    objective) is kept in the hard gate but EXCLUDED from the mean-%/worst-cell
    aggregate. (a) with the candidate feasible everywhere, fitness == the scored
    cell's score alone (n14 not averaged in); (b) candidate infeasible on the
    feasibility-only cell still FAILS the hard gate."""
    from funwake2.controller.cascade import _is_feasibility_only
    assert _is_feasibility_only("parque_n14_uniform"), "n14 must be feasibility_only"
    assert not _is_feasibility_only("dei_n50")

    cells = ["dei_n50", "parque_n14_uniform"]
    base = {"cells": {"dei_n50": {"seeds": {str(s): 5560.0 for s in range(5)}},
                      "parque_n14_uniform": {"seeds": {str(s): 184.8 for s in range(5)}}}}

    # (a) feasible everywhere -> fitness excludes n14
    ok = Cascade(_dry_cfg(tempfile.mkdtemp()), evaluate_fn=make_fake_eval(),
                 baselines=base)
    src = open(os.path.join(_SEED_DIR, "seed_cosine.py")).read()
    b = ok.stage_b(src, cells, [0, 1])
    assert b.passed, b.per_cell
    assert b.per_cell["parque_n14_uniform"]["feasibility_only"] is True
    assert b.per_cell["dei_n50"]["feasibility_only"] is False
    # aggregate == the single scored cell's score, NOT the mean of both
    assert abs(b.fitness - b.per_cell["dei_n50"]["score"]) < 1e-9, \
        (b.fitness, b.per_cell["dei_n50"]["score"], b.per_cell["parque_n14_uniform"]["score"])
    assert abs(b.worst_cell - b.per_cell["dei_n50"]["score"]) < 1e-9

    # (b) infeasible on the feasibility-only cell -> hard gate still fails
    bad = Cascade(_dry_cfg(tempfile.mkdtemp()),
                  evaluate_fn=make_fake_eval(infeasible_cells=["parque_n14_uniform"]),
                  baselines=base)
    b2 = bad.stage_b(src, cells, [0, 1])
    assert not b2.passed, "infeasible on a feasibility-only cell must still fail"
    return {"aggregate_excludes_n14": True,
            "n14_score_present": b.per_cell["parque_n14_uniform"]["score"],
            "hard_gate_still_applies": (not b2.passed)}


def test_farm_balanced_aggregate_parity():
    """Farm-balanced aggregate: fitness = mean over farms of the per-farm mean
    cell score. A uniform +1% lift across ALL of one farm's scored cells raises
    fitness by 1%/n_farms, the SAME for either farm, INDEPENDENT of how many cells
    each farm has (the training set has more DEI cells than Parque cells). This is
    the parity a flat mean over cells lacks (it would over-weight DEI)."""
    from funwake2.controller.cascade import _cell_farm
    cells = ["dei_n50", "dei_n80_omnidir", "parque_n20", "parque_n10_omnidir"]
    assert [_cell_farm(c) for c in cells] == ["dei", "dei", "parque", "parque"]
    base = {"cells": {c: {"seeds": {str(s): 1000.0 for s in range(5)}} for c in cells}}

    def make(pcts):  # deterministic eval: score_c == pcts[cell] (AEP vs ref 1000)
        def ev(cell, fn, seed=0, total_steps=8000, gamma_min=0.01):
            return {"cell": cell, "seed": seed, "steps": total_steps,
                    "gamma_min": gamma_min, "feasible": True,
                    "aep_gwh": 1000.0 * (1.0 + pcts[cell] / 100.0)}
        return ev

    src = open(os.path.join(_SEED_DIR, "seed_native.py")).read()
    dei_up = Cascade(_dry_cfg(tempfile.mkdtemp()), baselines=base,
                     evaluate_fn=make({"dei_n50": 1, "dei_n80_omnidir": 1,
                                       "parque_n20": 0, "parque_n10_omnidir": 0}))
    par_up = Cascade(_dry_cfg(tempfile.mkdtemp()), baselines=base,
                     evaluate_fn=make({"dei_n50": 0, "dei_n80_omnidir": 0,
                                       "parque_n20": 1, "parque_n10_omnidir": 1}))
    fa = dei_up.stage_b(src, cells, [0, 1, 2, 3, 4]).fitness
    fb = par_up.stage_b(src, cells, [0, 1, 2, 3, 4]).fitness
    assert abs(fa - fb) < 1e-9, (fa, fb)          # equal farm influence
    assert abs(fa - 0.5) < 1e-9 and abs(fb - 0.5) < 1e-9   # 1% / 2 farms
    # a single-farm +1% is NOT diluted by the other farm's cell count
    return {"dei_up_fitness": fa, "parque_up_fitness": fb, "parity": abs(fa - fb) < 1e-9}


def test_gbar_only_cell_pending_off_gbar():
    """A gbar-only capability-frontier cell (parque_n30_uniform) is PENDING off
    gbar (enable_stage_b_plus=False): deferred to the elite tier, never gating —
    even if it would be infeasible were it evaluated."""
    from funwake2.controller.cascade import _is_gbar_only
    assert _is_gbar_only("parque_n30_uniform")
    assert not _is_gbar_only("dei_n50")
    cells = ["dei_n50", "parque_n30_uniform"]
    base = {"cells": {"dei_n50": {"seeds": {str(s): 5560.0 for s in range(5)}}}}
    c = Cascade(_dry_cfg(tempfile.mkdtemp()), baselines=base,
                evaluate_fn=make_fake_eval(infeasible_cells=["parque_n30_uniform"]))
    src = open(os.path.join(_SEED_DIR, "seed_native.py")).read()
    b = c.stage_b(src, cells, [0, 1])
    assert b.per_cell["parque_n30_uniform"]["status"] == "PENDING"
    assert b.per_cell["parque_n30_uniform"]["gates"] is False
    assert b.passed, "a PENDING gbar-only cell must not gate off gbar"
    return {"pending": True, "passed_despite_infeasible_frontier": b.passed}


def test_stage_a_gross_filter_and_causes():
    """Stage A is a GROSS filter, not texture-floor-tight: a candidate 0.5% below
    the reference PASSES (it is not mass-rejected), one ~2% below is rejected as
    'below_ref', and an infeasible one is rejected as 'infeasible'. The `causes`
    tally is the pilot metric (rejection rate by cause)."""
    cells = ["dei_n50"]
    base = {"cells": {"dei_n50": {"seeds": {str(s): 1000.0 for s in range(5)}}}}
    src = open(os.path.join(_SEED_DIR, "seed_native.py")).read()

    def ev(frac_below, feasible=True):
        def f(cell, fn, seed=0, total_steps=8000, gamma_min=0.01):
            return {"cell": cell, "seed": seed, "steps": total_steps,
                    "gamma_min": gamma_min, "feasible": feasible,
                    "aep_gwh": 1000.0 * (1.0 - frac_below)}
        return f

    a = Cascade(_dry_cfg(tempfile.mkdtemp()), baselines=base,
                evaluate_fn=ev(0.005)).stage_a(src, cells, [0, 1])
    assert a.passed and a.causes["ok"] == 2 and a.causes["below_ref"] == 0, a.causes
    b = Cascade(_dry_cfg(tempfile.mkdtemp()), baselines=base,
                evaluate_fn=ev(0.02)).stage_a(src, cells, [0, 1])
    assert (not b.passed) and b.causes["below_ref"] == 2, b.causes
    c = Cascade(_dry_cfg(tempfile.mkdtemp()), baselines=base,
                evaluate_fn=ev(0.0, feasible=False)).stage_a(src, cells, [0, 1])
    assert (not c.passed) and c.causes["infeasible"] == 2, c.causes
    return {"ok_case": a.causes, "below_ref_case": b.causes, "infeasible_case": c.causes}


def test_fitness_scale_constant_with_infeasible_ref():
    # native reference infeasible on dei_n50; candidate feasible -> still scored.
    cfg = _dry_cfg(tempfile.mkdtemp())
    casc = Cascade(cfg, evaluate_fn=make_fake_eval(infeasible_ref="dei_n50"))
    src = open(os.path.join(_SEED_DIR, "seed_cosine.py")).read()
    b = casc.stage_b(src, ["dei_n50"], [0, 1])
    sc = b.per_cell["dei_n50"]
    assert sc["score"] != float("-inf") and sc["feasible"], sc
    assert isinstance(sc["base_mean"], float)
    return {"score_pct": round(sc["score"], 4), "base_is_scale_constant": True,
            "candidate_feasible": sc["feasible"]}


def test_firewall_stage_c():
    cfg = _dry_cfg(tempfile.mkdtemp())
    casc = Cascade(cfg, evaluate_fn=make_fake_eval(), baselines=_fake_baselines())
    src = open(os.path.join(_SEED_DIR, "seed_native.py")).read()
    c = casc.stage_c(src, "parque_n10_omnidir", [0], floor_gwh=0.1)
    from funwake2.controller.controller import _fw_stagec
    fw = _fw_stagec(c)
    assert "_firewalled" not in fw, "raw holdout AEP must not cross the firewall"
    assert "margin_over_floor" in fw and "responds_to_gamma_min" in fw
    return {"firewalled_keys_stripped": True, "exposed": sorted(fw.keys())}


def test_resume_bit_identity():
    tmp = tempfile.mkdtemp()
    fake = make_fake_eval()
    shared_cache = os.path.join(tmp, "cache")     # content-addressed, shared

    # clean run (state A) to completion — populates the shared cache
    cfgA = _dry_cfg(os.path.join(tmp, "A"), shared_cache)
    cA = Cascade(cfgA, evaluate_fn=fake, baselines=_fake_baselines())
    ctrlA = Controller(cfgA, MockEngine(), cA)
    assert ctrlA.run(max_generations=None) == "DONE"
    arcA = open(os.path.join(cfgA.state_dir, "archive.json"), "rb").read()
    linA = _lineage_core(cfgA.lineage_path)

    # resumed run (state B): ONE generation per call, fresh Controller each call
    # (simulating kill + resume). SAME content-addressed cache => every eval is
    # a cache HIT and nothing is recomputed.
    cfgB = _dry_cfg(os.path.join(tmp, "B"), shared_cache)
    recompute = []
    for _ in range(cfgB.generations + 3):
        c = Cascade(cfgB, evaluate_fn=fake, baselines=_fake_baselines())
        ctrlB = Controller(cfgB, MockEngine(), c)      # reloads state from disk
        st = ctrlB.run(max_generations=1)
        recompute.append(c.cache.misses)
        if st in ("DONE", "ABORT"):
            break
    arcB = open(os.path.join(cfgB.state_dir, "archive.json"), "rb").read()
    linB = _lineage_core(cfgB.lineage_path)

    assert arcA == arcB, "resumed archive must be byte-identical to the clean run"
    assert linA == linB, ("resumed lineage provenance must be identical "
                          "(modulo wall-clock timestamp/walltime)")
    assert all(m == 0 for m in recompute), \
        f"resume recomputed evals (shared cache should give 0 misses): {recompute}"
    return {"archive_bit_identical": True, "lineage_bit_identical": True,
            "recompute_misses_per_resume_call": recompute}


class _CrashEngine(MockEngine):
    """MockEngine that raises after `crash_after` successful mutate() calls —
    simulates a process kill MID-generation."""
    def __init__(self, crash_after):
        super().__init__()
        self.crash_after = crash_after
        self.n = 0

    def mutate(self, ctx):
        if self.n >= self.crash_after:
            raise KeyboardInterrupt("simulated mid-generation kill")
        self.n += 1
        return super().mutate(ctx)


def test_resume_midgen_crash_bit_identity():
    """Kill MID gen-2 (after the first proposal of the 2nd evolutionary
    generation), then resume -> archive + lineage bit-identical to an
    uninterrupted run, with cache hits (no recompute)."""
    tmp = tempfile.mkdtemp()
    fake = make_fake_eval()
    shared = os.path.join(tmp, "cache")

    # uninterrupted reference
    cfgA = _dry_cfg(os.path.join(tmp, "A"), shared)
    Controller(cfgA, MockEngine(), Cascade(cfgA, evaluate_fn=fake,
               baselines=_fake_baselines())).run(max_generations=None)
    arcA = open(os.path.join(cfgA.state_dir, "archive.json"), "rb").read()
    linA = _lineage_core(cfgA.lineage_path)

    # crashing run: gen0 has proposals_per_gen(2) mutations, then gen1 proposal0
    # is the 3rd mutate() -> crash on the 4th (mid gen-1). run() will seed + do
    # gen0 fully (checkpoint), then crash inside gen1's step_generation.
    cfgB = _dry_cfg(os.path.join(tmp, "B"), shared)
    crashed = False
    try:
        Controller(cfgB, _CrashEngine(crash_after=3),
                   Cascade(cfgB, evaluate_fn=fake, baselines=_fake_baselines())
                   ).run(max_generations=None)
    except KeyboardInterrupt:
        crashed = True
    assert crashed, "mid-gen crash was not triggered"

    # resume with a fresh, non-crashing controller (reloads gen-boundary state)
    misses = []
    for _ in range(5):
        c = Cascade(cfgB, evaluate_fn=fake, baselines=_fake_baselines())
        st = Controller(cfgB, MockEngine(), c).run(max_generations=1)
        misses.append(c.cache.misses)
        if st in ("DONE", "ABORT"):
            break
    arcB = open(os.path.join(cfgB.state_dir, "archive.json"), "rb").read()
    linB = _lineage_core(cfgB.lineage_path)

    assert arcA == arcB, "mid-gen resume archive must be byte-identical"
    assert linA == linB, "mid-gen resume lineage provenance must be identical"
    assert all(m == 0 for m in misses), f"resume recomputed evals: {misses}"
    return {"crashed_midgen": True, "archive_bit_identical": True,
            "lineage_bit_identical": True, "recompute_after_resume": misses}


def _lineage_core(path):
    """Reproducible provenance core of the lineage: every field EXCEPT the two
    inherently wall-clock fields (timestamp, walltime_s)."""
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            rec.pop("timestamp", None)
            rec.pop("walltime_s", None)
            out.append(rec)
    return json.dumps(out, sort_keys=True)


def _dry_cfg(root, cache_dir=None):
    return C.RunConfig(
        run_id="unit", state_dir=os.path.join(root, "state"),
        cache_dir=cache_dir or os.path.join(root, "cache"),
        lineage_path=os.path.join(root, "lineage.jsonl"),
        dry_run=True, dry_cells=["dei_n50", "parque_n20"],
        generations=2, proposals_per_gen=2, num_islands=2,
        stage_a_seeds=[0, 1], stage_b_seeds=[0, 1], stage_c_seeds=[0],
        max_usd=1000.0, max_tokens=10**9)


ALL = [test_archive_binning, test_cost_abort, test_cascade_order_and_fast_reject,
       test_fitness_scale_constant_with_infeasible_ref, test_firewall_stage_c,
       test_resume_bit_identity, test_resume_midgen_crash_bit_identity]

if __name__ == "__main__":
    ok = True
    for t in ALL:
        try:
            res = t()
            print(f"PASS {t.__name__}: {res}")
        except AssertionError as e:
            ok = False
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            ok = False
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)
