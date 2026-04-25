"""Aggregate Experiment B: 2x2 scaffold x interface contingency.

Classifies each of the 4 cells (existing CLI runs + new structured-output
runs) into "discovery" or "reimplementation" based on the dominant motif
in the best feasible scripts. Heuristic, manual override possible via
classification_overrides.json.

Discovery markers:    bump, restart, cyclic-betas, alpha-squeeze
Reimplementation:     SLSQP, scipy.optimize.minimize(method='SLSQP'), or
                      a thin wrapper around topfarm_sgd_solve.

Usage:
    pixi run python experiments/B_scaffold_cross/aggregate.py
"""
import json
import os
import re


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR = os.path.dirname(os.path.abspath(__file__))


CELLS = [
    # (label, scaffold, interface, run_dir)
    ("Claude/CLI/sched",         "cli", "narrow", "results_agent_schedule_only_5hr"),
    ("Claude/CLI/full",          "cli", "broad",  "results_agent_claude_fullopt"),
    ("Claude/SO/sched",          "so",  "narrow", "results_agent_claude_anthropic_api_sched"),
    ("Claude/SO/full",           "so",  "broad",  "results_agent_claude_anthropic_api_fullopt"),
]


REIMPL_PAT = re.compile(r"SLSQP|topfarm_sgd_solve|method\s*=\s*['\"]SLSQP['\"]")
DISCOVERY_PAT = re.compile(r"bump|restart|cosine_restart|cyclic|alpha_squeeze|n_cycles")


def classify(script_path):
    if not script_path or not os.path.exists(script_path):
        return "no_data"
    src = open(script_path).read()
    if REIMPL_PAT.search(src):
        return "reimplementation"
    if DISCOVERY_PAT.search(src):
        return "discovery"
    return "neither"


def best_script(run_dir):
    log_path = os.path.join(run_dir, "attempt_log.json")
    if not os.path.exists(log_path):
        return None
    log = json.load(open(log_path))
    feas = [e for e in log if e.get("train_feasible") and "train_aep" in e]
    if not feas:
        return None
    best = max(feas, key=lambda e: e["train_aep"])
    return os.path.join(run_dir, f"iter_{best['attempt']:03d}.py")


def main():
    overrides_path = os.path.join(EXP_DIR, "classification_overrides.json")
    overrides = json.load(open(overrides_path)) if os.path.exists(overrides_path) else {}

    contingency = {"cli": {"narrow": None, "broad": None},
                    "so": {"narrow": None, "broad": None}}
    rows = []
    for label, scaffold, interface, dir_rel in CELLS:
        run_dir = os.path.join(PROJECT_ROOT, dir_rel)
        script = best_script(run_dir)
        cls = overrides.get(label) or classify(script)
        rows.append({"label": label, "scaffold": scaffold,
                     "interface": interface, "run_dir": dir_rel,
                     "best_script": script, "classification": cls})
        contingency[scaffold][interface] = cls

    summary = {"contingency": contingency, "rows": rows}
    out = os.path.join(EXP_DIR, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Quick interpretation
    cli = contingency["cli"]
    so = contingency["so"]
    if cli == so:
        print("\n[B] Scaffold-invariant: outcome depends on interface only.")
    else:
        print("\n[B] Scaffold-sensitive: scaffold and interface both matter.")


if __name__ == "__main__":
    main()
