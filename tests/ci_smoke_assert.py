#!/usr/bin/env python
"""Assertion helper for the CI smoke workflow (.github/workflows/smoke.yml).

Parses the JSON that FunWake's scorer / test tools print to stdout and checks
the advertised quickstart invariants. Keeps the assertions in Python (typed,
tolerant) instead of brittle bash grepping.

Usage (pipe the tool's stdout in):
    python tools/run_optimizer.py results/seed_schedule.py --schedule-only \
        | python tests/ci_smoke_assert.py --check seed
    python tools/run_optimizer.py runs/schedule_only_5hr/iter_192.py --schedule-only \
        | python tests/ci_smoke_assert.py --check iter192
    python tools/run_tests.py results/seed_schedule.py --quick \
        | python tests/ci_smoke_assert.py --check tests
    python tools/test_generalization.py runs/schedule_only_5hr/iter_192.py --schedule-only \
        | python tests/ci_smoke_assert.py --check generalization

Determinism (compare two saved scorer JSON outputs):
    python tests/ci_smoke_assert.py --check determinism --files a.json b.json

Exits 0 on success, 1 (with a message) on any failed assertion.
"""
import argparse
import json
import sys

# Documented quickstart values (README "Quick start"). Sources:
#   seed AEP 5529.2 / baseline 5540.72 -> results/baselines.json farm "1"
SEED_AEP = 5529.2
BASELINE = 5540.72
# Absolute seed-AEP tolerance is generous: XLA/BLAS reduction order differs
# across CPU architectures (this repo sees ~1.7 GWh arm64<->x86 drift on the
# same code+seed), and 5529.2 was measured on arm64. The load-bearing checks
# are structural (feasible flags; iter_192 beats the baseline by ~21 GWh).
SEED_TOL = 5.0
# Determinism is same-machine (two runs on one runner) -> should be near-exact.
DET_TOL = 0.5


def _load_stdin_json():
    """Return the last JSON object printed to stdin (tools may print logs first)."""
    raw = sys.stdin.read()
    obj = None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
    if obj is None:
        _fail(f"no JSON object found in stdin; raw output was:\n{raw[:1000]}")
    return obj


def _ok(msg):
    print(f"[ci-assert] PASS: {msg}")


def _fail(msg):
    print(f"[ci-assert] FAIL: {msg}")
    sys.exit(1)


def check_seed(d):
    if d.get("feasible") is not False:
        _fail(f"seed should be infeasible by design, got feasible={d.get('feasible')}")
    aep = d.get("aep_gwh")
    if aep is None or abs(aep - SEED_AEP) > SEED_TOL:
        _fail(f"seed AEP {aep} not within {SEED_TOL} of {SEED_AEP}")
    if abs(d.get("baseline", 0) - BASELINE) > 0.01:
        _fail(f"baseline {d.get('baseline')} != {BASELINE}")
    _ok(f"seed AEP={aep} feasible=False baseline={d.get('baseline')}")


def check_iter192(d):
    if d.get("feasible") is not True:
        _fail(f"iter_192 should be feasible, got feasible={d.get('feasible')}")
    aep, base = d.get("aep_gwh"), d.get("baseline")
    if aep is None or base is None or aep <= base:
        _fail(f"iter_192 AEP {aep} must exceed baseline {base}")
    _ok(f"iter_192 AEP={aep} > baseline={base} (gap {d.get('gap')})")


def check_tests(d):
    if d.get("passed") is not True:
        _fail(f"run_tests reported passed={d.get('passed')} (seed xfail should keep it green)")
    _ok("run_tests passed (seed feasibility is expected-fail)")


def check_generalization(d):
    if d.get("passed") is not True or d.get("feasible") is not True:
        _fail(f"generalization failed: passed={d.get('passed')} feasible={d.get('feasible')} issues={d.get('issues')}")
    _ok("iter_192 feasible on held-out ROWP")


def check_determinism(files):
    vals = []
    for f in files:
        with open(f) as fh:
            vals.append(json.load(fh).get("aep_gwh"))
    if any(v is None for v in vals):
        _fail(f"missing aep_gwh in determinism inputs: {vals}")
    spread = max(vals) - min(vals)
    if spread > DET_TOL:
        _fail(f"determinism spread {spread:.4f} GWh > {DET_TOL} across {vals}")
    _ok(f"determinism spread {spread:.4f} GWh <= {DET_TOL} ({vals})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", required=True,
                   choices=["seed", "iter192", "tests", "generalization", "determinism"])
    p.add_argument("--files", nargs="+", help="JSON files for --check determinism")
    a = p.parse_args()

    if a.check == "determinism":
        if not a.files or len(a.files) < 2:
            _fail("--check determinism needs at least two --files")
        check_determinism(a.files)
        return

    d = _load_stdin_json()
    {"seed": check_seed, "iter192": check_iter192,
     "tests": check_tests, "generalization": check_generalization}[a.check](d)


if __name__ == "__main__":
    main()
