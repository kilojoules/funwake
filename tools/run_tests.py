#!/usr/bin/env python
"""Run unit tests on an optimizer module. Wraps playground/test_optimizer.py.

Usage:
    python tools/run_tests.py <optimizer.py> [problem.json] [--quick]

Prints JSON result to stdout.
"""
import json
import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: run_tests.py <optimizer.py> [problem.json] [--quick]"}))
        sys.exit(1)

    script = sys.argv[1]
    args = sys.argv[2:]

    test_runner = os.path.join(os.path.dirname(__file__), "..", "playground", "test_optimizer.py")
    cmd = [sys.executable, test_runner, os.path.abspath(script)] + args

    env = {**os.environ, "JAX_ENABLE_X64": "True"}
    pixwake = os.path.join(os.path.dirname(__file__), "..", "playground", "pixwake", "src")
    env["PYTHONPATH"] = f"{pixwake}:{env.get('PYTHONPATH', '')}"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                                cwd=os.path.join(os.path.dirname(__file__), ".."), env=env)
        output = result.stdout.strip()
        passed = result.returncode == 0
        print(json.dumps({
            "passed": passed,
            "output": output,
            "errors": result.stderr[-500:] if result.stderr else None,
        }))
    except subprocess.TimeoutExpired:
        print(json.dumps({"passed": False, "output": "Tests timed out after 180s"}))


if __name__ == "__main__":
    main()
