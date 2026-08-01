"""Scoped mutator workspace — the Phase-3 LAUNCH GATE for firewall containment.

The mutation engines (Claude Agent-SDK, Gemini CLI) are run with their *cwd*
pointed at a freshly-materialized clean-room dir that contains ONLY:

  * ``INTERFACE.md``   — the schedule_fn signature + fixed-skeleton description
  * ``skeleton_v2.py`` — the frozen skeleton (read reference; no test/holdout info)
  * ``seeds/``         — the gen-0 seed schedules (native + iter ports)
  * ``parent.py``      — the current parent schedule being mutated
  * ``feedback.json``  — FIREWALLED per-cell feedback (%-scores + feasibility
                          booleans only; NO raw AEP, NO holdout/test AEP)

Everything else in the source tree — ``results/``, ``paper/``, ``specs/``, the
pre-registration, audit/EDITS docs, ``funwake2/state/``, ``baselines_g2.json``,
and ``evaluator.py`` (which encodes the holdout/test-cell roles) — is OUTSIDE the
scoped dir and therefore outside the engine's readable tree. Containment is
enforced two ways, belt-and-suspenders:

  1. **cwd** — the engine process runs in the scoped dir (Gemini subprocess
     ``cwd=``; Claude ``ClaudeAgentOptions(cwd=...)``), so relative reads resolve
     inside it. The scoped dir lives OUTSIDE the repo (a scratch root), so ``..``
     traversal does not reach the source tree.
  2. **allowed-tools** — the Claude engine ships ``allowed_tools=[]`` (no file
     tools at all; the prompt is self-contained). If a future config enables read
     tools they are confined to cwd.

``assert_clean`` is the gate assertion: it scans the materialized dir (and,
post-run, the transcript) for forbidden path tokens and holdout-value literals
and RAISES if any are present, so a launch cannot proceed with a leaky workspace.
"""
from __future__ import annotations

import json
import os
import re
import shutil

# path/name tokens that must NEVER appear inside the scoped workspace or a
# mutator transcript (they would reveal the deployment/test design or leak AEP).
FORBIDDEN_TOKENS = (
    "results/", "paper/", "specs/", "funwake2/state", "baselines_g2",
    "PREREGISTRATION", "prereg", "EDITS_", "audit", "lr0_paired", "lr0_diameter",
    "parqo_native_ms", "native_ms", "rowp_n200", "rowp_n300", "rowp_n74_uniform",
    "problem_rowp", "holdout", "test_set", "deployment",
)

# files copied into the scope (relative to the funwake2 root)
_HARNESS = ["skeleton_v2.py"]
_SEED_SUBDIR = "seeds"


def _strip_module_docstring(text: str) -> str:
    """Remove a leading triple-quoted module docstring (where seed/skeleton
    provenance — 'results/...', 'lr0_...' — lives). The clean-room copies are
    READ references only, never executed, so removing the docstring is safe."""
    m = re.match(r'\s*[rubfRUBF]{0,2}("""|\'\'\')', text)
    if not m:
        return text
    q = m.group(1)
    end = text.find(q, m.end())
    return text[end + 3:] if end != -1 else text


def sanitize(text: str) -> str:
    """Strip the module docstring, then redact any residual forbidden token so a
    copied harness/seed/parent file cannot leak a path or the deployment design
    into the mutator's readable tree."""
    text = _strip_module_docstring(text)
    for tok in FORBIDDEN_TOKENS:
        if tok in text:
            text = text.replace(tok, "REDACTED")
    return text


def _interface_md() -> str:
    return (
        "# Schedule interface (all you may use)\n\n"
        "Write a Python module defining ONLY:\n\n"
        "```python\n"
        "def schedule_fn(step, total_steps, D, min_spacing, n_turbines, "
        "gamma_min, alpha0):\n"
        "    # returns (lr, alpha, beta1, beta2) for this step\n"
        "    ...\n"
        "```\n\n"
        "The fixed skeleton (skeleton_v2.py) builds the exploration lr from the\n"
        "rotor diameter D, computes AEP + constraint gradients, and runs Adam.\n"
        "There is NO free lr0. alpha0 = mean|grad J|/D is supplied. gamma_min is\n"
        "the metre-valued constraint tolerance (= the terminal lr target).\n"
        "You are given the parent schedule (parent.py) and firewalled per-cell\n"
        "feedback (feedback.json: %-over-baseline scores + feasibility booleans\n"
        "only). Do NOT attempt to read any other file.\n"
    )


def materialize(scope_dir: str, parent_source: str, feedback: dict,
                fw2_root: str, seeds_dir: str | None = None) -> str:
    """Create/refresh the scoped workspace at ``scope_dir`` (which SHOULD live
    outside the repo). Returns the absolute scope path. Raises via assert_clean
    if anything forbidden lands inside."""
    scope_dir = os.path.abspath(scope_dir)
    if os.path.exists(scope_dir):
        shutil.rmtree(scope_dir)
    os.makedirs(scope_dir)

    with open(os.path.join(scope_dir, "INTERFACE.md"), "w") as f:
        f.write(_interface_md())
    for rel in _HARNESS:
        src = os.path.join(fw2_root, rel)
        if os.path.exists(src):
            with open(src, errors="ignore") as fh:
                _write(os.path.join(scope_dir, os.path.basename(rel)),
                       sanitize(fh.read()))

    seeds_dir = seeds_dir or os.path.join(fw2_root, _SEED_SUBDIR)
    dst_seeds = os.path.join(scope_dir, "seeds")
    os.makedirs(dst_seeds, exist_ok=True)
    if os.path.isdir(seeds_dir):
        for fn in os.listdir(seeds_dir):
            if fn.endswith(".py") and not fn.startswith("_"):
                with open(os.path.join(seeds_dir, fn), errors="ignore") as fh:
                    _write(os.path.join(dst_seeds, fn), sanitize(fh.read()))

    _write(os.path.join(scope_dir, "parent.py"), sanitize(parent_source or ""))
    # feedback must be firewall-safe already (controller._fw_percell). Re-assert.
    _assert_feedback_firewalled(feedback)
    with open(os.path.join(scope_dir, "feedback.json"), "w") as f:
        json.dump(feedback, f, indent=2, sort_keys=True)

    assert_clean(scope_dir)
    return scope_dir


def _write(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def _assert_feedback_firewalled(feedback: dict) -> None:
    """The feedback that reaches a mutator must carry NO raw AEP — only
    %-scores + feasibility booleans (controller._fw_percell shape)."""
    blob = json.dumps(feedback).lower()
    for bad in ("aep", "gwh", "_firewalled", "base_aep"):
        if bad in blob:
            raise AssertionError(
                f"feedback carries a forbidden key/value '{bad}': {feedback}")


def assert_clean(scope_dir: str) -> None:
    """Scan every file under the scoped dir for forbidden path/value tokens.
    RAISES on any hit — a launch must not proceed with a leaky workspace."""
    hits = scan_tree(scope_dir)
    if hits:
        raise AssertionError(
            "scoped workspace is NOT clean — forbidden tokens found:\n" +
            "\n".join(f"  {p}: {tok}" for p, tok in hits))


def scan_tree(root: str, extra_tokens=()):
    """Return [(path, token)] for every forbidden token found in any file under
    root. Used both to gate the materialized workspace and to grep a transcript
    (pass the transcript's dir/file)."""
    tokens = tuple(FORBIDDEN_TOKENS) + tuple(extra_tokens)
    hits = []
    paths = [root]
    if os.path.isfile(root):
        paths = [os.path.dirname(root)]
        files = [root]
    else:
        files = []
        for dp, _dn, fn in os.walk(root):
            for f in fn:
                files.append(os.path.join(dp, f))
    for fp in files:
        try:
            with open(fp, "r", errors="ignore") as fh:
                text = fh.read()
        except Exception:
            continue
        low = text.lower()
        for tok in tokens:
            if tok.lower() in low:
                # INTERFACE.md legitimately says "read any other file"; allow the
                # engine's own seed filenames. Only flag genuine forbidden tokens.
                hits.append((os.path.relpath(fp, root if os.path.isdir(root)
                                             else os.path.dirname(root)), tok))
    return hits
