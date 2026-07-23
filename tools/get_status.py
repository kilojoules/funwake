#!/usr/bin/env python
"""Get current agent status. Prints JSON summary to stdout.

Usage:
    python tools/get_status.py [--log attempt_log.json] [--baselines baselines.json] [--train-farm 1]
"""
import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default=None,
                   help="Path to attempt_log.json (searches results_agent_* dirs if not set)")
    p.add_argument("--baselines", default="results/baselines.json")
    p.add_argument("--train-farm", default="1")
    args = p.parse_args()

    attempts = []
    if os.path.exists(args.log):
        with open(args.log) as f:
            attempts = json.load(f)

    baseline = 0
    try:
        with open(args.baselines) as f:
            baselines = json.load(f)
        baseline = baselines.get(args.train_farm, {}).get("aep_gwh", 0)
    except FileNotFoundError:
        pass

    successes = [a for a in attempts if "train_aep" in a]
    errors = [a for a in attempts if "error" in a]
    best_aep = max((a["train_aep"] for a in successes), default=0)

    summary = {
        "attempts": len(attempts),
        "successes": len(successes),
        "errors": len(errors),
        "best_aep": round(best_aep, 2),
        "baseline": round(baseline, 2),
        "gap": round(best_aep - baseline, 2),
    }

    # Equivalent-cost SGD reference: run_optimizer.py attaches these fields
    # only when scoring the training problem, so older logs (or non-training
    # scorings) simply lack them — tolerate that.
    equiv_attempts = [a for a in successes
                      if a.get("equiv_cost_sgd") is not None]
    if equiv_attempts:
        equiv_ref = equiv_attempts[-1]["equiv_cost_sgd"]
        summary["equiv_cost_sgd"] = round(equiv_ref, 2)
        summary["gap_equiv"] = round(best_aep - equiv_ref, 2)

    # If any attempts used --stress-problem, surface adversarial-selection state.
    stress_attempts = [a for a in successes if "stress_aep" in a]
    if stress_attempts:
        feasible_both = [a for a in stress_attempts
                          if a.get("train_feasible") and a.get("stress_feasible")
                          and a.get("min_gap") is not None]
        summary["stress_attempts"] = len(stress_attempts)
        summary["stress_feasible_both"] = len(feasible_both)
        if feasible_both:
            best_min = max(feasible_both, key=lambda a: a["min_gap"])
            summary["best_min_gap"] = round(best_min["min_gap"], 2)
            summary["best_min_gap_attempt"] = best_min["attempt"]
            summary["best_min_gap_train_aep"] = best_min["train_aep"]
            summary["best_min_gap_stress_aep"] = best_min["stress_aep"]

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
