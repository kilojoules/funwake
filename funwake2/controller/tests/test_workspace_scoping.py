"""Launch-gate test: the scoped mutator workspace contains ONLY harness + seeds
+ firewalled feedback, and assert_clean RAISES on any forbidden path/holdout
token or non-firewalled feedback. Runs with nothing installed (no jax/LLM)."""
import json
import os
import sys
import tempfile

_THIS = os.path.dirname(os.path.abspath(__file__))
_FW2 = os.path.dirname(os.path.dirname(_THIS))
_ROOT = os.path.dirname(_FW2)
for _p in (_ROOT, _FW2):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from funwake2.controller import workspace as W  # noqa: E402

_FW2_ROOT = _FW2  # funwake2/


def _mk(scope, feedback=None):
    fb = feedback if feedback is not None else {
        "dei_n50": {"score_pct": 0.12, "feasible": True},
        "parque_n20": {"score_pct": -0.03, "feasible": True},
    }
    return W.materialize(scope, parent_source="def schedule_fn(*a): return 1,1,.1,.2\n",
                         feedback=fb, fw2_root=_FW2_ROOT)


def test_scope_contains_only_allowed():
    with tempfile.TemporaryDirectory() as d:
        scope = _mk(os.path.join(d, "scope"))
        entries = set(os.listdir(scope))
        assert "INTERFACE.md" in entries
        assert "parent.py" in entries
        assert "feedback.json" in entries
        assert "seeds" in entries and os.path.isdir(os.path.join(scope, "seeds"))
        # NOTHING that encodes the deployment/test design may be present
        for forbidden in ("evaluator.py", "results", "paper", "specs",
                          "PREREGISTRATION.md", "baselines_g2.json", "state"):
            assert forbidden not in entries, f"{forbidden} leaked into scope"
        # feedback is firewall-safe: %-scores + booleans, no AEP
        fb = json.load(open(os.path.join(scope, "feedback.json")))
        assert all(set(v) <= {"score_pct", "feasible"} for v in fb.values())
    return {"ok": True}


def test_assert_clean_raises_on_forbidden_token():
    with tempfile.TemporaryDirectory() as d:
        scope = _mk(os.path.join(d, "scope"))
        # inject a leak: a stray file that names a forbidden path
        with open(os.path.join(scope, "leak.txt"), "w") as f:
            f.write("peek at results/funwake2_prereg/PREREGISTRATION.md\n")
        raised = False
        try:
            W.assert_clean(scope)
        except AssertionError:
            raised = True
        assert raised, "assert_clean must RAISE when a forbidden token is present"
    return {"raised": True}


def test_feedback_with_raw_aep_rejected():
    with tempfile.TemporaryDirectory() as d:
        raised = False
        try:
            _mk(os.path.join(d, "scope"),
                feedback={"rowp_n74": {"aep_gwh": 4258.7, "feasible": True}})
        except AssertionError:
            raised = True
        assert raised, "materialize must reject feedback carrying raw AEP"
    return {"raised": True}


def test_scan_tree_flags_holdout_value_in_transcript():
    with tempfile.TemporaryDirectory() as d:
        tpath = os.path.join(d, "transcript.txt")
        with open(tpath, "w") as f:
            f.write("model reasoning ... the rowp_n200 holdout looked like ...\n")
        hits = W.scan_tree(tpath)
        assert hits, "scan_tree must flag a forbidden token in a transcript"
    return {"hits": True}
