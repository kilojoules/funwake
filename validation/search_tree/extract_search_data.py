"""Extract per-iteration (agent, attempt, axis, hypothesis, aep, feasible) records
from Claude + Gemini schedule-only 5hr runs.

Also computes per-attempt cumulative wall-time within sessions (gaps > 30 min
treated as off-time and skipped — the runs were resumed after quota / matrix
re-eval windows). Filters output to attempts within the first 5 hr of
session-time per agent so the comparison is budget-matched.

Output: validation/search_tree/search_data.json — list of dicts ready for plotting.

AXIS classification is heuristic (no HYPOTHESIS/AXIS docstring tags exist in
either run — agents ignored STRATEGY_REGISTRY_RULE).
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUNS = [
    ("Claude", os.path.join(ROOT, "runs/schedule_only_5hr")),
    ("Gemini", os.path.join(ROOT, "runs/gemini_cli_5hr")),
]

OFF_TIME_GAP_S = 30 * 60  # gaps longer than 30 min = off-time (resume)


AXIS_PATTERNS = [
    ("dual_bump",          r"dual[- ]?bump|two bumps|bumps at .* and|bump1|bump2"),
    ("single_bump",        r"\bbump\b|single bump|gaussian bump"),
    ("warm_restart",       r"sgdr|warm restart|restarts?\b|cosine restart"),
    ("cyclic_lr",          r"cyclic lr|cyclical lr|sinusoidal lr|sin\(.*step|cycle of \d|cycles?\b"),
    ("noise_shake",        r"shake|noise injection|sinusoidal.*lr|sinusoidal.*per|jitter"),
    ("warmup_cosine",      r"warmup.*cosine|warm-up.*cosine|cosine.*warmup"),
    ("cosine_anneal",      r"cosine ann|cosine decay|cosine schedule|cos\(.*pi"),
    ("piecewise_lr",       r"piecewise|piece-wise|step decay|step-decay|linear pieces"),
    ("exp_decay_lr",       r"exponential decay|exp decay|exp\(-.*step"),
    ("poly_decay_lr",      r"polynomial decay|poly decay|inverse.time|1/\(1\+"),
    ("alpha_squeeze",      r"alpha.{0,8}squeeze|terminal.{0,8}alpha|alpha.*ramp|alpha quadratic|alpha.*surge|quad.*alpha"),
    ("alpha_dip",          r"alpha.{0,8}dip|penalty dip|dip.{0,8}alpha"),
    ("cyclic_alpha",       r"cyclic alpha|alpha cycle|sinusoidal alpha"),
    ("alpha_adaptive",     r"adaptive alpha|alpha.*gradient|grad.*alpha|magnitude.*alpha"),
    ("beta_phase",         r"beta.*phase|phase.*beta|beta.*ramping|ramp.*beta|early.*late.*beta"),
    ("beta_cyclic",        r"cyclic beta|beta.*cycle"),
    ("beta_zero",          r"beta1\s*=\s*0\b|beta_1\s*=\s*0\b|no momentum|zero momentum"),
    ("beta_standard",      r"standard adam|adam.*0\.9.*0\.999|beta1\s*=\s*0\.9"),
    ("beta_topfarm",       r"topfarm.*beta|beta.*0\.1.*0\.2|beta1\s*=\s*0\.1"),
    ("coupled_alpha",      r"coupled alpha|alpha.*=.*alpha0.*lr|alpha.*1/lr|inverse.{0,5}lr.*alpha"),
    ("monotonic_alpha",    r"monotonic alpha|linear alpha|increasing alpha"),
]


def classify_axis(text: str) -> str:
    text_low = text.lower()
    for axis, pat in AXIS_PATTERNS:
        if re.search(pat, text_low):
            return axis
    return "other"


def read_script(dirpath, attempt_n):
    path = os.path.join(dirpath, f"iter_{attempt_n:03d}.py")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        text = f.read()
    m = re.search(r'"""(.+?)"""', text[:1500], re.DOTALL)
    docstring = m.group(1).strip() if m else ""
    return docstring, text


def session_elapsed_s(entries):
    """For each entry in order of attempt, return cumulative within-session wall-time
    (gaps > OFF_TIME_GAP_S are skipped, so the value is 'time the agent was actually
    running')."""
    entries = sorted(entries, key=lambda e: e["attempt"])
    prev_ts = None
    cum = 0.0
    out = []
    for e in entries:
        ts = e.get("timestamp")
        if ts is None:
            out.append(None)
            continue
        if prev_ts is not None:
            dt = ts - prev_ts
            if dt > OFF_TIME_GAP_S:
                dt = 0.0  # treat as resume — no time accrued
            cum += max(0.0, dt)
        prev_ts = ts
        out.append(cum)
    return out


def main():
    out = []
    for agent, dirpath in RUNS:
        log_path = os.path.join(dirpath, "attempt_log.json")
        if not os.path.exists(log_path):
            print(f"SKIP {agent}: {log_path} not found")
            continue
        entries = json.load(open(log_path))
        # Dedup by attempt
        seen, deduped = set(), []
        for e in sorted(entries, key=lambda x: x["attempt"]):
            if e["attempt"] in seen: continue
            seen.add(e["attempt"])
            deduped.append(e)

        elapsed = session_elapsed_s(deduped)

        for i, e in enumerate(deduped):
            n = e["attempt"]
            docstring, full_text = read_script(dirpath, n)
            if docstring is None:
                axis, hypothesis = "other", "(script missing)"
            else:
                axis = classify_axis(docstring + "\n" + (full_text or ""))
                hypothesis = next(
                    (line.strip() for line in docstring.splitlines() if line.strip()),
                    "(no docstring)"
                )[:140]
            rec = {
                "agent": agent,
                "attempt": n,
                "elapsed_s": elapsed[i],
                "elapsed_hr": (elapsed[i] / 3600.0) if elapsed[i] is not None else None,
                "timestamp": e.get("timestamp"),
                "train_aep": e.get("train_aep"),
                "train_feasible": bool(e.get("train_feasible", False)),
                "rowp_aep": e.get("rowp_aep"),
                "rowp_feasible": bool(e.get("rowp_feasible", False)),
                "axis": axis,
                "hypothesis": hypothesis,
            }
            out.append(rec)

    # Stats
    from collections import Counter
    for agent, _ in RUNS:
        recs = [r for r in out if r["agent"] == agent]
        in_5hr = [r for r in recs if r["elapsed_hr"] is not None and r["elapsed_hr"] <= 5.0]
        print(f"{agent}: {len(recs)} total, {len(in_5hr)} within 5hr session-time")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_data.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path} ({len(out)} records)")


if __name__ == "__main__":
    main()
