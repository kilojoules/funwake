"""Evaluate a set of LLM-generated schedule scripts on all 48 cells
of the farm × N × wind-rose matrix. Writes per-cell (script, AEP,
feasibility) rows to results/matrix/schedules_matrix.json.

Usage:
    pixi run python tools/eval_matrix_schedules.py
"""
import json
import os
import subprocess
import sys
import time


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


SCHEDULES = [
    # (label, script_path)
    ("Claude schedule (iter 192)",
     "results_agent_schedule_only_5hr/iter_192.py"),
    ("Gemini schedule",
     "results_agent_gemini_cli_5hr/iter_192.py"),
]


def score_one(script_rel, problem_rel, timeout=180):
    env = {
        **os.environ,
        "JAX_ENABLE_X64": "True",
        "PYTHONPATH": (
            os.path.join(PROJECT_ROOT, "playground", "pixwake", "src")
            + ":" + os.environ.get("PYTHONPATH", "")
        ),
    }
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "tools", "run_optimizer.py"),
        os.path.join(PROJECT_ROOT, script_rel),
        "--problem", os.path.join(PROJECT_ROOT, problem_rel),
        "--timeout", str(timeout),
        "--log", "/dev/null",
        "--schedule-only",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout + 60, env=env,
                           cwd=PROJECT_ROOT)
        return json.loads(r.stdout)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)[:200]}


def main():
    manifest = json.load(
        open(os.path.join(PROJECT_ROOT, "results", "matrix", "manifest.json")))
    out_path = os.path.join(PROJECT_ROOT, "results", "matrix",
                            "schedules_matrix.json")

    # Resume-safe: load existing, skip cells we already have
    existing = {}
    if os.path.exists(out_path):
        existing = json.load(open(out_path))

    results = dict(existing)
    t_start = time.time()
    total = len(SCHEDULES) * len(manifest["cells"])
    done = 0

    for label, script in SCHEDULES:
        for cell in manifest["cells"]:
            done += 1
            key = f"{label}|{cell['farm']}_n{cell['n']}_rose{cell['rose']}"
            if key in results and "aep_gwh" in results[key]:
                continue
            t0 = time.time()
            r = score_one(script, cell["path"], timeout=180)
            elapsed = time.time() - t0
            entry = {
                "label": label,
                "script": script,
                "farm": cell["farm"],
                "n": cell["n"],
                "rose": cell["rose"],
                "time_s": round(elapsed, 1),
            }
            entry.update(r)
            results[key] = entry
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            total_elapsed = time.time() - t_start
            remaining = total - done
            etamin = (remaining * elapsed) / 60 if elapsed > 0 else 0
            msg = (f"[{done}/{total}] {label} @ {cell['farm']} "
                   f"n={cell['n']} rose={cell['rose']}: ")
            if "aep_gwh" in r:
                feas = "✓" if r.get("feasible") else "✗"
                msg += f"AEP={r['aep_gwh']:.1f} {feas} ({elapsed:.0f}s, ETA~{etamin:.0f}m)"
            else:
                msg += f"ERR {r.get('error','')[:60]}"
            print(msg, flush=True)

    print()
    print(f"Total elapsed: {(time.time()-t_start)/60:.1f} min")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
