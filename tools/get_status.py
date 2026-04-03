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
    p.add_argument("--log", default="results_agent/attempt_log.json")
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

    print(json.dumps({
        "attempts": len(attempts),
        "successes": len(successes),
        "errors": len(errors),
        "best_aep": round(best_aep, 2),
        "baseline": round(baseline, 2),
        "gap": round(best_aep - baseline, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
