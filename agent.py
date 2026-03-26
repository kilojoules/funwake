#!/usr/bin/env python
"""FunSearch agent for layout optimization.

Multi-turn: the LLM writes an optimizer script, it runs on training farms
(1-9), scores feed back, the LLM refines. The held-out test farm (0) is
only evaluated at the very end.

The LLM has full access to its pixwake clone. It can read source, modify
code, use topfarm_sgd_solve as a building block, or write from scratch.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TRAIN_FARMS = [1]
TEST_FARM = 0


# ── LLM ────────────────────────────────────────────────────────────────

def llm_generate(prompt: str, model: str, api_key: str,
                  temperature: float = 0.7,
                  api_base: str | None = None,
                  timeout: float | None = None,
                  provider: str = "together") -> str:
    """Generate via Together AI, Gemini, or any OpenAI-compatible endpoint.

    Parameters
    ----------
    timeout : float | None
        Maximum seconds to wait for the LLM response. None means no limit.
    provider : str
        "together", "gemini", or "openai" (for vLLM / OpenAI-compatible).
    """
    if provider == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            max_output_tokens=8192,
            temperature=temperature,
        )
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        return resp.text
    elif api_base or provider == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key=api_key or "none",
                        timeout=timeout)
    else:
        from together import Together
        client = Together(api_key=api_key, timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8192,
        temperature=temperature,
    )
    return resp.choices[0].message.content


# ── Safety ──────────────────────────────────────────────────────────────

_BLOCKED = [
    "import subprocess", "from subprocess",
    "import socket", "from socket",
    "import http", "from http",
    "import urllib", "from urllib",
    "import requests", "from requests",
    "os.system(", "os.popen(", "os.exec",
    "__import__(", "getattr(__builtins__",
    "eval(", "exec(", "compile(",
    "import shutil", "from shutil",
    "import ctypes", "from ctypes",
]


def safety_check(code: str) -> tuple[bool, str]:
    """Deterministic safety check: blocklist + syntax validation.

    No LLM involved — avoids false-positive rejections that waste iterations.
    """
    for token in _BLOCKED:
        if token in code:
            return False, f"Blocked: {token!r}"
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    return True, "OK"


# ── Error diagnosis ────────────────────────────────────────────────────

_ERROR_FIXES = [
    ("from jax.config import config",
     'WRONG: `from jax.config import config` does not exist in modern JAX.\n'
     'CORRECT: `import jax; jax.config.update("jax_enable_x64", True)`'),
    ("from jax import config",
     'WRONG: `from jax import config` does not exist.\n'
     'CORRECT: `import jax; jax.config.update("jax_enable_x64", True)`'),
    ("name 'jnp' is not defined",
     'You forgot `import jax.numpy as jnp`. Add it after `import jax`.'),
    ("name 'np' is not defined",
     'You forgot `import numpy as np`.'),
    # Wrong top-level imports from pixwake
    (re.compile(r"cannot import name '(?:topfarm_sgd_solve|SGDSettings|boundary_penalty|spacing_penalty)' from 'pixwake'"),
     'WRONG import path. These are NOT in the pixwake top-level package.\n'
     'CORRECT:\n'
     '  from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve\n'
     '  from pixwake.optim.sgd import boundary_penalty, spacing_penalty'),
    (re.compile(r"cannot import name 'BastankhahGaussianDeficit' from 'pixwake'"),
     'WRONG import path. BastankhahGaussianDeficit is NOT in pixwake top-level.\n'
     'CORRECT: `from pixwake.deficit import BastankhahGaussianDeficit`'),
    ("from pixwake.bastankhah",
     'WRONG module path. There is no pixwake.bastankhah module.\n'
     'CORRECT: `from pixwake.deficit import BastankhahGaussianDeficit`'),
    ("No module named 'pixwake.optim.boundary_penalty'",
     'WRONG module path.\n'
     'CORRECT: `from pixwake.optim.sgd import boundary_penalty, spacing_penalty`'),
    # Timeout
    ("Timeout after",
     'Your script timed out. It must complete within the time limit per farm.\n'
     'The baseline runs in ~10-30s. Reduce iterations or complexity.\n'
     'Avoid scipy.optimize.differential_evolution or other global optimizers\n'
     'with large populations — they are too slow for this problem.'),
    # Common runtime errors
    ("Some entries in x0 lay outside the specified bounds",
     'scipy bounds error: initial positions are outside the optimizer bounds.\n'
     'If using scipy, derive bounds from the actual boundary_vertices polygon,\n'
     'not hardcoded values. Or use topfarm_sgd_solve which handles boundaries\n'
     'via penalty functions.'),
    ("os.environ[\"FUNWAKE_OUTPUT\"] =",
     'WRONG: Do NOT assign to os.environ["FUNWAKE_OUTPUT"].\n'
     'WRITE to the file at that path:\n'
     '  with open(os.environ["FUNWAKE_OUTPUT"], "w") as f:\n'
     '      json.dump({"x": [...], "y": [...]}, f)'),
]


def _diagnose_error(error_text: str) -> str:
    """Match error text against known mistakes and return corrective hints."""
    hints = []
    for pattern, fix in _ERROR_FIXES:
        if isinstance(pattern, re.Pattern):
            if pattern.search(error_text):
                hints.append(fix)
        elif pattern in error_text:
            hints.append(fix)
    return "\n\n".join(hints)


# ── Script runner ───────────────────────────────────────────────────────

def run_on_farm(code: str, farm_id: int, playground: Path, results_dir: Path,
                timeout_s: int = 300, run_id: str = "") -> dict:
    """Run LLM's optimizer script on one farm. Returns {aep, time, ...}."""
    playground = playground.resolve()
    results_dir = results_dir.resolve()

    script_name = f"_generated_optimizer_{run_id}.py" if run_id else "_generated_optimizer.py"
    script_path = playground / script_name
    script_path.write_text(code)

    output_path = results_dir / f"_llm_farm{farm_id}.json"
    problem_path = results_dir / f"problem_farm{farm_id}.json"
    output_path.unlink(missing_ok=True)

    if not problem_path.exists():
        return {"error": f"Missing {problem_path}"}

    pixwake_src = str((playground / "pixwake" / "src").resolve())
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{pixwake_src}:{existing}" if existing else pixwake_src
    env["JAX_ENABLE_X64"] = "True"
    env["FUNWAKE_OUTPUT"] = str(output_path)
    env["FUNWAKE_PROBLEM"] = str(problem_path)

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=timeout_s,
            cwd=str(playground), env=env)
    except subprocess.TimeoutExpired:
        return {"error": f"Timeout after {timeout_s}s", "farm_id": farm_id}
    elapsed = time.time() - t0

    if result.returncode != 0:
        stderr = result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr
        return {"error": f"rc={result.returncode}:\n{stderr}",
                "farm_id": farm_id, "time": elapsed}

    if not output_path.exists():
        return {"error": "No output written",
                "stdout": result.stdout[-500:],
                "farm_id": farm_id, "time": elapsed}

    with open(output_path) as f:
        layout = json.load(f)

    return {"x": layout["x"], "y": layout["y"],
            "farm_id": farm_id, "time": elapsed,
            "stdout": result.stdout[-300:]}


def score_layout(layout: dict, farm_id: int, benchmark_script: Path,
                 wind_csv: str, playground: Path = None) -> dict:
    """Score via firewalled benchmark (clean pixwake)."""
    benchmark_script = benchmark_script.resolve()
    tmp = benchmark_script.parent / f"_tmp_layout_{farm_id}.json"
    with open(tmp, "w") as f:
        json.dump(layout, f)
    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "True"
    # Benchmark scorer needs pixwake in PYTHONPATH
    if playground:
        pixwake_src = str((playground / "pixwake" / "src").resolve())
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{pixwake_src}:{existing}" if existing else pixwake_src
    result = subprocess.run(
        [sys.executable, str(benchmark_script),
         "--wind-csv", wind_csv, "score",
         "--farm-id", str(farm_id), "--layout", str(tmp)],
        capture_output=True, text=True, timeout=120, env=env,
        cwd=str(benchmark_script.parent.parent))
    tmp.unlink(missing_ok=True)
    if result.returncode != 0:
        return {"error": result.stderr[-500:]}
    for line in result.stdout.splitlines():
        if "AEP=" in line:
            aep = float(line.split("AEP=")[1].split("GWh")[0].strip())
            feas = "feasible=True" in line
            return {"aep_gwh": aep, "feasible": feas}
    return {"error": f"Parse error: {result.stdout}"}


# ── Prompts ─────────────────────────────────────────────────────────────

def _format_farm_problems(results_dir: Path, train_farms: list[int]) -> str:
    """Load all training farm problem JSONs and format a summary for the LLM."""
    sections = []
    for fid in train_farms:
        p = results_dir / f"problem_farm{fid}.json"
        if not p.exists():
            continue
        with open(p) as f:
            info = json.load(f)
        bv = info["boundary_vertices"]
        bv_str = ", ".join(f"({v[0]:.1f}, {v[1]:.1f})" for v in bv)
        sections.append(
            f"### Farm {fid} — {info.get('farm_name', '')}\n"
            f"  n_target: {info['n_target']}, "
            f"min_spacing: {info['min_spacing_m']}m, "
            f"rotor_diameter: {info['rotor_diameter']}m\n"
            f"  boundary ({len(bv)} vertices): [{bv_str}]\n"
            f"  init_x ({len(info['init_x'])} turbines): "
            f"[{', '.join(f'{v:.1f}' for v in info['init_x'][:5])}, ...]\n"
            f"  init_y: [{', '.join(f'{v:.1f}' for v in info['init_y'][:5])}, ...]"
        )
    return "\n\n".join(sections)


def _read_source(playground: Path) -> str:
    snippets = []
    # (file, max_lines) — sgd.py needs more lines to include
    # boundary_penalty (line ~179) and spacing_penalty (line ~201)
    source_files = [
        ("pixwake/src/pixwake/optim/sgd.py", 250),
        ("pixwake/src/pixwake/core.py", 100),
        ("pixwake/src/pixwake/optim/boundary.py", 212),
    ]
    for rp, max_lines in source_files:
        p = playground / rp
        if p.exists():
            lines = p.read_text().splitlines()[:max_lines]
            snippets.append(f"# ── {rp} (first {max_lines} lines) ──\n" + "\n".join(lines))
    return "\n\n".join(snippets)


# The baseline working script — used as fallback reference when no iteration
# has succeeded yet, and appended as a correct-imports block on every ITERATE.
_BASELINE_TEMPLATE = """\
import jax
jax.config.update("jax_enable_x64", True)
import os, json
import jax.numpy as jnp
import numpy as np
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve

# Load problem
with open(os.environ["FUNWAKE_PROBLEM"]) as f:
    info = json.load(f)

# Setup
D = info["rotor_diameter"]
ws_arr = jnp.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25.0])
power = jnp.array([0,0,2.399,209.258,689.198,1480.608,2661.238,4308.929,6501.057,9260.516,12081.404,13937.297,14705.016,14931.039,14985.209,14996.906,14999.343,14999.855,14999.966,14999.992,14999.998,14999.999,15000,15000,15000,15000.0])
ct = jnp.array([0.889,0.889,0.889,0.8,0.8,0.8,0.8,0.8,0.8,0.793,0.735,0.61,0.476,0.37,0.292,0.234,0.191,0.158,0.132,0.112,0.096,0.083,0.072,0.063,0.055,0.049])
turbine = Turbine(rotor_diameter=D, hub_height=150.0,
                  power_curve=Curve(ws=ws_arr, values=power),
                  ct_curve=Curve(ws=ws_arr, values=ct))
sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))

wd = jnp.array(info["wind_rose"]["directions_deg"])
ws = jnp.array(info["wind_rose"]["speeds_ms"])
weights = jnp.array(info["wind_rose"]["weights"])
boundary = jnp.array(info["boundary_vertices"])
init_x = jnp.array(info["init_x"])
init_y = jnp.array(info["init_y"])
min_spacing = info["min_spacing_m"]

def objective(x, y):
    r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
    p = r.power()[:, :len(x)]
    return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

settings = SGDSettings(learning_rate=50.0, max_iter=500,
                       additional_constant_lr_iterations=500, tol=1e-6)
opt_x, opt_y = topfarm_sgd_solve(objective, init_x, init_y,
                                  boundary, min_spacing, settings)

with open(os.environ["FUNWAKE_OUTPUT"], "w") as f:
    json.dump({"x": [float(v) for v in opt_x],
               "y": [float(v) for v in opt_y]}, f)
"""


CORRECT_IMPORTS = """\
## CORRECT import paths (do NOT change these)
```python
import jax
jax.config.update("jax_enable_x64", True)
import os, json
import jax.numpy as jnp
import numpy as np
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve, boundary_penalty, spacing_penalty
```
"""


SYSTEM = """\
You are an optimization researcher. Write a Python script that optimizes
a wind farm layout to MAXIMIZE AEP (Annual Energy Production).

## Problem

~50 turbines (IEA 15 MW, D=240m) inside a polygon boundary. Minimum
spacing 4×D (960m). Differentiable wake simulation via pixwake (JAX).
No neighbor interference — pure layout optimization.

Your script will be evaluated on a single training farm boundary. A
held-out farm with the same shape is used for final validation.

## Constraints

Your output layout MUST satisfy two hard constraints. The scorer checks
these EXACTLY as described below — layouts that violate them are penalized
or marked infeasible.

### 1. Boundary constraint
Every turbine must be INSIDE the polygon defined by `boundary_vertices`
(convex, CCW order). Feasibility threshold: `boundary_penalty(x, y,
boundary) < 1e-3`.

`boundary_penalty` (from `pixwake.optim.sgd`) computes:
- For each polygon edge, the signed distance from every turbine to that
  edge line (positive = inside / left of CCW edge, negative = outside).
- Per turbine: min over all edges → the most-violated edge.
- Violations: `min(0, min_distance)` — only negative (outside) values.
- Penalty: `sum(violations ** 2)`.

### 2. Spacing constraint
Every pair of turbines must be at least `min_spacing` apart (= 4×D =
960m). Feasibility threshold: `min_pairwise_distance >= min_spacing * 0.99`
(i.e. >= 950.4m).

The scorer computes all pairwise Euclidean distances:
```
dx = x[:, None] - x[None, :]
dy = y[:, None] - y[None, :]
dist = sqrt(dx**2 + dy**2 + eye(n)*1e10)   # diagonal masked
min_dist = min(dist)
```

### Penalty functions available for optimization
You can import and use these directly:
```python
from pixwake.optim.sgd import boundary_penalty, spacing_penalty
# boundary_penalty(x, y, boundary_vertices) -> scalar (0 = all inside)
# spacing_penalty(x, y, min_spacing)        -> scalar (0 = all spaced)
```

`spacing_penalty` computes: `sum(max(0, min_spacing**2 - d**2))` for all
unique pairs where `d**2 < min_spacing**2`.

Both are JAX-differentiable. `topfarm_sgd_solve` combines them with
KS aggregation and an increasing penalty coefficient alpha.

## Baseline to beat

topfarm_sgd_solve — constrained ADAM with LR decay and KS penalties.
Average baseline AEP across training farms: {avg_baseline:.2f} GWh.

Per-farm baselines:
{farm_baselines}

## Your script must

1. `import jax; jax.config.update("jax_enable_x64", True)` at the top
2. Load problem from `os.environ["FUNWAKE_PROBLEM"]` (JSON with
   init_x, init_y, boundary_vertices, wind_rose, n_target, etc.)
3. Set up pixwake wake simulation
4. Optimize the layout
5. Write {{"x": [...], "y": [...]}} to `os.environ["FUNWAKE_OUTPUT"]`
6. Ensure constraints are satisfied (boundary_penalty < 1e-3, min
   pairwise distance >= 950.4m)

## Available

pixwake (WakeSimulation, Curve, Turbine, BastankhahGaussianDeficit,
topfarm_sgd_solve, SGDSettings, boundary_penalty, spacing_penalty),
jax, numpy, scipy

## Training farm details

These are ALL the training farms your script will be evaluated on.
Each has a different polygon boundary but the same wind rose and turbine.

{farm_problems}

## Key pixwake source
{source_excerpts}
"""

FIRST = """\
Here is a WORKING template that scores ~5527 GWh (the baseline). Modify
it to do better. The API usage below is CORRECT — do not change import
paths or function signatures.

```python
""" + _BASELINE_TEMPLATE + """\
```

Modify this to beat the baseline. Ideas: try different learning rates,
multistart with random initializations, different beta1/beta2, or write
your own optimizer entirely.

Write the COMPLETE script.
"""

ITERATE = """\
## Iteration {i}

Per-farm AEP this iteration vs baseline:
{per_farm}

Average AEP: {avg_aep:.2f} GWh (baseline: {avg_baseline:.2f}, gap: {gap:+.2f})
Best average so far: {best_avg:.2f} GWh (iter {best_i})

{feedback}

## BEST SCRIPT SO FAR (iter {best_i}, {best_avg:.2f} GWh avg)

This script WORKS — correct imports, correct API. Build on it.
Do NOT change import paths or function call signatures.

```python
{best_code}
```

Write an improved COMPLETE script. Keep the working imports and API calls.
Try a genuinely different optimization strategy, not just tweaking numbers.
"""


def _extract_code(text: str) -> str:
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


# ── Main loop ───────────────────────────────────────────────────────────

def run(playground_dir: str, benchmark_script: str, wind_csv: str,
        results_dir: str, baselines: dict,
        n_iters: int, model: str, api_key: str,
        train_farms: list[int] = None, timeout_s: int = 300,
        output_dir: str = "results", temperature: float = 0.7,
        run_id: str = "", llm_timeout: float | None = None,
        probe_timeout_s: int | None = None, provider: str = "together"):

    playground = Path(playground_dir)
    benchmark = Path(benchmark_script)
    res_dir = Path(results_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if train_farms is None:
        train_farms = TRAIN_FARMS
    if probe_timeout_s is None:
        probe_timeout_s = min(120, timeout_s)

    # Baseline summary
    baseline_aeps = {fid: baselines[str(fid)]["aep_gwh"] for fid in train_farms}
    avg_baseline = np.mean(list(baseline_aeps.values()))
    farm_baseline_str = "\n".join(
        f"  Farm {fid}: {baseline_aeps[fid]:.2f} GWh" for fid in train_farms)

    source = _read_source(playground)
    farm_problems = _format_farm_problems(res_dir, train_farms)
    system = SYSTEM.format(
        avg_baseline=avg_baseline,
        farm_baselines=farm_baseline_str,
        source_excerpts=source,
        farm_problems=farm_problems,
    )

    history = []
    best_avg = 0.0
    best_iter = -1
    consecutive_failures = 0

    for i in range(n_iters):
        print(f"\n{'='*60}  ITERATION {i}  {'='*60}")

        # Prompt
        if i == 0:
            prompt = system + "\n\n" + FIRST
        else:
            last = history[-1] if history else {}
            per_farm_str = ""
            if "farms" in last:
                for fid, r in sorted(last["farms"].items()):
                    bl = baseline_aeps.get(int(fid), 0)
                    aep = r.get("aep", 0)
                    diff = aep - bl
                    per_farm_str += f"  Farm {fid}: {aep:.2f} GWh (baseline {bl:.2f}, {diff:+.2f})\n"

            if "error" in last or last.get("n_success", 9) < 5:
                err_detail = ""
                if "farms" in last:
                    for fid, r in sorted(last["farms"].items()):
                        if "error" in r:
                            err_detail = r["error"][:1500]
                            break
                if not err_detail and "error" in last:
                    err_detail = str(last["error"])[:1500]

                # Diagnose known errors and provide explicit corrections
                hint = _diagnose_error(err_detail)
                if hint:
                    feedback = (
                        f"Last attempt had errors:\n```\n{err_detail}\n```\n\n"
                        f"## KNOWN FIX\n{hint}\n\n"
                        f"Use EXACTLY the import paths shown in the CORRECT "
                        f"imports section below."
                    )
                else:
                    feedback = (
                        f"Last attempt had errors:\n```\n{err_detail}\n```\n"
                        f"Fix the error. Keep the same import paths as the "
                        f"best script."
                    )
            elif last.get("avg_aep", 0) <= best_avg and i > 1:
                feedback = "No improvement. Try a fundamentally different approach."
            else:
                feedback = "Progress. Keep refining."

            # After 2+ consecutive failures, escalate: reset to baseline
            if consecutive_failures >= 2:
                feedback = (
                    f"WARNING: {consecutive_failures} consecutive failures. "
                    f"You are stuck in a loop.\n"
                    f"RESET: Start from the WORKING BASELINE below. Make ONLY "
                    f"small, incremental changes. Do NOT rewrite the imports "
                    f"or API calls — they are correct as written.\n\n"
                    f"{feedback}"
                )

            # Load best script so far — fall back to baseline template
            best_code_path = out / "best_optimizer.py"
            if best_code_path.exists():
                best_code = best_code_path.read_text()
            else:
                best_code = _BASELINE_TEMPLATE

            prompt = system + "\n\n" + ITERATE.format(
                i=i, per_farm=per_farm_str or "  (errors on all farms)",
                avg_aep=last.get("avg_aep", 0), avg_baseline=avg_baseline,
                gap=last.get("avg_aep", 0) - avg_baseline,
                best_avg=best_avg, best_i=best_iter,
                feedback=feedback, best_code=best_code,
            ) + "\n\n" + CORRECT_IMPORTS

        # Generate
        print(f"Generating...{f' (timeout {llm_timeout}s)' if llm_timeout else ''}")
        t0 = time.time()
        try:
            raw = llm_generate(prompt, model=model, api_key=api_key,
                               temperature=temperature, timeout=llm_timeout,
                               provider=provider)
        except Exception as e:
            print(f"  LLM error: {e}")
            history.append({"i": i, "error": str(e)})
            consecutive_failures += 1
            continue
        code = _extract_code(raw)
        print(f"  {len(code)} chars, {time.time()-t0:.1f}s")
        (out / f"iter_{i:03d}.py").write_text(code)

        # Safety (deterministic — no LLM call)
        print("Safety check...")
        safe, reason = safety_check(code)
        if not safe:
            print(f"  REJECTED: {reason}")
            history.append({"i": i, "error": f"Safety: {reason}"})
            consecutive_failures += 1
            continue
        print(f"  {reason}")

        # Run on each training farm (with fail-fast and probe timeout)
        farm_results = {}
        aeps = []
        farm_consecutive_errors = 0
        for idx, fid in enumerate(train_farms):
            print(f"  Farm {fid}...", end=" ", flush=True)
            # First farm uses shorter probe timeout to detect slow scripts
            this_timeout = probe_timeout_s if idx == 0 else timeout_s
            r = run_on_farm(code, fid, playground, res_dir, this_timeout,
                           run_id=run_id)
            if "error" in r:
                print(f"ERROR: {str(r['error'])[:300]}")
                farm_results[str(fid)] = {"error": str(r["error"])[:2000]}
                farm_consecutive_errors += 1
                is_timeout = "Timeout" in str(r.get("error", ""))
                # Fail fast: skip remaining farms after 2 consecutive errors,
                # or immediately if the first farm (probe) times out
                if is_timeout and idx == 0:
                    print(f"  [probe timeout] First farm timed out at "
                          f"{this_timeout}s — skipping remaining farms")
                    break
                if farm_consecutive_errors >= 2:
                    print(f"  [fail-fast] {farm_consecutive_errors} consecutive "
                          f"errors — skipping remaining farms")
                    break
                continue

            farm_consecutive_errors = 0

            # Score via firewalled benchmark
            sc = score_layout({"x": r["x"], "y": r["y"]},
                              fid, benchmark, wind_csv, playground)
            if "error" in sc:
                print(f"Score error: {sc['error'][:100]}")
                farm_results[str(fid)] = {"error": sc["error"][:300]}
                farm_consecutive_errors += 1
                if farm_consecutive_errors >= 2:
                    print(f"  [fail-fast] {farm_consecutive_errors} consecutive "
                          f"errors — skipping remaining farms")
                    break
                continue

            farm_consecutive_errors = 0
            aep = sc["aep_gwh"]
            bl = baseline_aeps[fid]
            diff = aep - bl
            print(f"{aep:.2f} GWh ({diff:+.2f} vs baseline)")
            farm_results[str(fid)] = {"aep": aep, "time": r["time"]}
            aeps.append(aep)

        if aeps:
            avg_aep = np.mean(aeps)
            print(f"\n  Average: {avg_aep:.2f} GWh "
                  f"({avg_aep - avg_baseline:+.2f} vs baseline)")
        else:
            avg_aep = 0.0
            print("\n  All farms failed.")

        entry = {"i": i, "avg_aep": avg_aep, "farms": farm_results,
                 "n_success": len(aeps)}

        if avg_aep > 0:
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        if avg_aep > best_avg:
            best_avg = avg_aep
            best_iter = i
            entry["is_best"] = True
            print(f"  *** NEW BEST: {avg_aep:.2f} GWh ***")
            # Save best script
            (out / "best_optimizer.py").write_text(code)
        history.append(entry)

        # Write history incrementally so plots can read in-progress runs
        with open(out / "history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)

    # ── Final: evaluate best on held-out test farm ──
    print(f"\n{'='*60}")
    print("HELD-OUT TEST: Farm 0 (dk0w_tender_3)")
    print(f"{'='*60}")
    best_code = (out / "best_optimizer.py").read_text() if (out / "best_optimizer.py").exists() else None
    if best_code:
        r = run_on_farm(best_code, TEST_FARM, playground, res_dir, timeout_s,
                        run_id=run_id)
        if "error" not in r:
            sc = score_layout({"x": r["x"], "y": r["y"]},
                              TEST_FARM, benchmark, wind_csv, playground)
            if "error" not in sc:
                test_bl = baselines.get(str(TEST_FARM), {}).get("aep_gwh", 0)
                test_aep = sc["aep_gwh"]
                print(f"  Test AEP: {test_aep:.2f} GWh "
                      f"(baseline: {test_bl:.2f}, gap: {test_aep - test_bl:+.2f})")
                history.append({"test": True, "aep": test_aep,
                                "baseline": test_bl})
            else:
                print(f"  Test score error: {sc['error'][:300]}")
        else:
            print(f"  Test run error: {str(r['error'])[:300]}")

    # Save history
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"\nHistory saved to {out / 'history.json'}")


def main():
    p = argparse.ArgumentParser(description="FunSearch: LLM writes optimizers")
    p.add_argument("--playground-dir", default="playground")
    p.add_argument("--benchmark-script", default="benchmarks/dei_layout.py")
    p.add_argument("--wind-csv", required=True)
    p.add_argument("--results-dir", default="results",
                   help="Dir with problem_farm*.json and baselines.json")
    p.add_argument("--n-iters", type=int, default=10)
    p.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    p.add_argument("--provider", default="together",
                   choices=["together", "gemini", "openai"],
                   help="LLM provider (default: together)")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--probe-timeout", type=int, default=None,
                   help="Timeout for first farm probe (default: min(120, timeout))")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--llm-timeout", type=float, default=None,
                   help="Max seconds for the LLM to generate a script (default: no limit)")
    p.add_argument("--run-id", default="",
                   help="Unique ID for concurrent runs (avoids playground conflicts)")
    args = p.parse_args()

    # Load API key based on provider
    key_files = {
        "together": "~/.together",
        "gemini": "~/.gem",
        "openai": "~/.openai",
    }
    key_path = os.path.expanduser(key_files.get(args.provider, "~/.together"))
    api_key = open(key_path).read().strip()

    with open(Path(args.results_dir) / "baselines.json") as f:
        baselines = json.load(f)

    run(
        playground_dir=args.playground_dir,
        benchmark_script=args.benchmark_script,
        wind_csv=args.wind_csv,
        results_dir=args.results_dir,
        baselines=baselines,
        n_iters=args.n_iters,
        model=args.model,
        api_key=api_key,
        timeout_s=args.timeout,
        output_dir=args.output_dir,
        temperature=args.temperature,
        run_id=args.run_id,
        llm_timeout=args.llm_timeout,
        probe_timeout_s=args.probe_timeout,
        provider=args.provider,
    )


if __name__ == "__main__":
    main()
