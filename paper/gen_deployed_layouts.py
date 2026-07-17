"""Generate iter_192 (deployed dual-bump) layouts for the anatomy figure.

Runs the schedule-only harness on 6 problems and stores the optimized
positions in paper/figs/deployed_layouts.json:
  dei_50   playground/problem.json            (training cell)
  rowp_74  results/problem_rowp.json          (held-out cell)
  dei_30 / dei_300 / rowp_30 / rowp_300       (matrix cells, own rose)
"""
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "results_agent_schedule_only_5hr/iter_192.py")
HARNESS = os.path.join(ROOT, "playground/harness.py")

JOBS = {
    "dei_50":   "playground/problem.json",
    "rowp_74":  "results/problem_rowp.json",
    "dei_30":   "results/matrix/problem_dei_n30_rosedei.json",
    "dei_300":  "results/matrix/problem_dei_n300_rosedei.json",
    "rowp_30":  "results/matrix/problem_rowp_n30_roserowp.json",
    "rowp_300": "results/matrix/problem_rowp_n300_roserowp.json",
}

OUT = os.path.join(ROOT, "paper/figs/deployed_layouts.json")
layouts = {}
if os.path.exists(OUT):
    layouts = json.load(open(OUT))

env_base = {
    **os.environ,
    "PYTHONPATH": os.path.join(ROOT, "playground/pixwake/src")
                  + ":" + os.path.join(ROOT, "playground"),
    "JAX_ENABLE_X64": "True",
    "JAX_PLATFORMS": "cpu",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}

for key, prob in JOBS.items():
    if key in layouts:
        print(f"{key}: cached, skip", flush=True)
        continue
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_path = tf.name
    env = {**env_base,
           "FUNWAKE_PROBLEM": os.path.join(ROOT, prob),
           "FUNWAKE_OUTPUT": out_path}
    r = subprocess.run([sys.executable, HARNESS, SCRIPT, "--schedule-only"],
                       env=env, cwd=ROOT, capture_output=True, text=True,
                       timeout=3600)
    if r.returncode != 0:
        print(f"{key}: FAILED\n{r.stderr[-400:]}", flush=True)
        continue
    lay = json.load(open(out_path))
    os.unlink(out_path)
    layouts[key] = lay
    with open(OUT, "w") as f:
        json.dump(layouts, f)
    print(f"{key}: {len(lay['x'])} turbines in {time.time()-t0:.0f}s", flush=True)

print("done:", sorted(layouts.keys()))
