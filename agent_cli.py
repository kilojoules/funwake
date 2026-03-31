#!/usr/bin/env python
"""Agentic layout optimizer — Claude Code-style tool-use loop.

The LLM gets tools to explore the codebase, write optimizer scripts,
run them on the training farm, and check results. It has a time budget
(default 10 min) to prototype freely. After time's up, the best script
is evaluated on held-out test cases.

Usage:
    pixi run python agent_cli.py \
        --wind-csv ~/clusters/energy_island_10y_daily_av_wind.csv \
        --provider gemini --model gemini-2.5-flash \
        --time-budget 600
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


# ── Safety ──────────────────────────────────────────────────────────────

_BLOCKED = [
    # Process/shell
    "import subprocess", "from subprocess",
    "os.system(", "os.popen(", "os.exec",
    "os.spawn",
    # Network
    "import socket", "from socket",
    "import http", "from http",
    "import urllib", "from urllib",
    "import requests", "from requests",
    "import xmlrpc", "from xmlrpc",
    "import ftplib", "from ftplib",
    "import smtplib", "from smtplib",
    # Code injection
    "__import__(", "getattr(__builtins__",
    "eval(", "exec(", "compile(",
    # Filesystem manipulation
    "import shutil", "from shutil",
    "os.remove(", "os.unlink(", "os.rmdir(",
    "os.rename(", "os.replace(",
    "os.makedirs(", "os.mkdir(",
    "pathlib.Path.unlink", "pathlib.Path.rmdir",
    # Dangerous native access
    "import ctypes", "from ctypes",
    "import signal", "from signal",
    "import pty", "from pty",
    "import resource", "from resource",
    # Env snooping
    "os.environ[",  # only FUNWAKE_PROBLEM and FUNWAKE_OUTPUT via open()
    "os.getenv(",
]

# These os.environ patterns are allowed (the script needs to read them)
_ALLOWED_ENV = [
    'os.environ["FUNWAKE_PROBLEM"]',
    "os.environ['FUNWAKE_PROBLEM']",
    'os.environ["FUNWAKE_OUTPUT"]',
    "os.environ['FUNWAKE_OUTPUT']",
]


def safety_check(code: str) -> tuple[bool, str]:
    for token in _BLOCKED:
        if token in code:
            # Allow whitelisted os.environ access
            if token.startswith("os.environ"):
                if any(allowed in code for allowed in _ALLOWED_ENV):
                    # Check there are no OTHER os.environ usages
                    stripped = code
                    for a in _ALLOWED_ENV:
                        stripped = stripped.replace(a, "")
                    if "os.environ" not in stripped and "os.getenv" not in stripped:
                        continue
            return False, f"Blocked: {token!r}"
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    return True, "OK"


# ── Sandbox ─────────────────────────────────────────────────────────────

# Minimal env whitelist — only what the optimizer needs
_ENV_WHITELIST = {
    "PATH", "HOME", "TMPDIR", "USER", "LANG",
    "PYTHONPATH", "JAX_ENABLE_X64",
    "FUNWAKE_OUTPUT", "FUNWAKE_PROBLEM",
    # JAX/XLA internals
    "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE",
    "JAX_PLATFORMS", "JAX_TRACEBACK_FILTERING",
}


def _sandbox_env(playground: Path, output_path: Path,
                 problem_path: Path) -> dict:
    """Build a minimal, sanitized environment for the optimizer subprocess."""
    pixwake_src = str((playground / "pixwake" / "src").resolve())
    env = {}
    for k in _ENV_WHITELIST:
        if k in os.environ:
            env[k] = os.environ[k]
    env["PYTHONPATH"] = pixwake_src
    env["JAX_ENABLE_X64"] = "True"
    env["FUNWAKE_OUTPUT"] = str(output_path)
    env["FUNWAKE_PROBLEM"] = str(problem_path)
    # Force tmpdir into a sandboxed location
    env["TMPDIR"] = str(playground / "_tmp")
    env["HOME"] = str(playground / "_tmp")  # prevent ~/.* access
    return env


def _sandbox_profile(playground: Path, results_dir: Path) -> str:
    """Generate a macOS sandbox-exec profile for the optimizer subprocess.

    Allows: read from anywhere (Python, libs), write only to playground/_tmp
    and the results dir. Denies: all network access.
    """
    playground_s = str(playground.resolve())
    results_s = str(results_dir.resolve())
    tmpdir = str((playground / "_tmp").resolve())
    return f"""\
(version 1)
(allow default)
(deny network*)
(deny file-write*
    (require-not
        (require-any
            (subpath "{tmpdir}")
            (subpath "{results_s}")
            (subpath "{playground_s}")
            (subpath "/private/tmp")
            (subpath "/private/var")
            (subpath "/dev")
        )
    )
)
"""


# ── Script runner ───────────────────────────────────────────────────────

def run_on_farm(code: str, farm_id: int, playground: Path, results_dir: Path,
                timeout_s: int = 300, run_id: str = "") -> dict:
    playground = playground.resolve()
    results_dir = results_dir.resolve()

    # Ensure sandbox tmpdir exists
    tmpdir = playground / "_tmp"
    tmpdir.mkdir(exist_ok=True)

    script_name = f"_generated_optimizer_{run_id}.py" if run_id else "_generated_optimizer.py"
    script_path = playground / script_name
    script_path.write_text(code)

    output_path = results_dir / f"_llm_farm{farm_id}.json"
    # Support both "problem_farm1.json" and "problem_rowp.json" naming
    problem_path = results_dir / f"problem_farm{farm_id}.json"
    if not problem_path.exists():
        problem_path = results_dir / f"problem_{farm_id}.json"
    output_path.unlink(missing_ok=True)

    if not problem_path.exists():
        return {"error": f"Missing {problem_path}"}

    env = _sandbox_env(playground, output_path, problem_path)

    # Write sandbox profile
    profile_path = tmpdir / "_sandbox.sb"
    profile_path.write_text(_sandbox_profile(playground, results_dir))

    # Run inside macOS sandbox
    cmd = [
        "/usr/bin/sandbox-exec", "-f", str(profile_path),
        sys.executable, str(script_path),
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            cwd=str(playground), env=env)
    except subprocess.TimeoutExpired:
        return {"error": f"Timeout after {timeout_s}s"}
    elapsed = time.time() - t0

    if result.returncode != 0:
        stderr = result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr
        return {"error": f"rc={result.returncode}:\n{stderr}", "time": elapsed}

    if not output_path.exists():
        return {"error": "No output written", "stdout": result.stdout[-500:],
                "time": elapsed}

    with open(output_path) as f:
        layout = json.load(f)

    return {"x": layout["x"], "y": layout["y"], "time": elapsed,
            "stdout": result.stdout[-300:]}


def score_layout(layout: dict, farm_id: int, benchmark_script: Path,
                 wind_csv: str, playground: Path) -> dict:
    benchmark_script = benchmark_script.resolve()
    tmp = benchmark_script.parent / f"_tmp_layout_{farm_id}.json"
    with open(tmp, "w") as f:
        json.dump(layout, f)
    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "True"
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


# ── Tool definitions for the LLM ────────────────────────────────────────

def _score_on_rowp(code: str, playground: Path, results_dir: Path,
                   timeout_s: int, run_id: str) -> dict | None:
    """Silently run optimizer on ROWP and score. Returns None on any failure."""
    rowp_problem = results_dir / "problem_rowp.json"
    if not rowp_problem.exists():
        return None
    r = run_on_farm(code, "rowp", playground, results_dir, timeout_s, run_id)
    if "error" in r:
        return {"error": r["error"][:500]}
    try:
        pixwake_src = str((playground / "pixwake" / "src").resolve())
        if pixwake_src not in sys.path:
            sys.path.insert(0, pixwake_src)
        bench_dir = str(playground.parent / "benchmarks")
        if bench_dir not in sys.path:
            sys.path.insert(0, bench_dir)
        from dei_layout import ProblemBenchmark
        bm = ProblemBenchmark(str(rowp_problem))
        aep = bm.score(r["x"], r["y"])
        feas = bm.check_feasibility(r["x"], r["y"])
        return {
            "aep_gwh": aep,
            "feasible": feas["spacing_ok"] and feas["boundary_ok"],
            "time": r["time"],
        }
    except Exception as e:
        return {"error": str(e)[:500]}


def make_tools(playground: Path, results_dir: Path, benchmark: Path,
               wind_csv: str, baselines: dict, train_farm: int,
               timeout_s: int, run_id: str, out_dir: Path):
    """Create tool declarations and dispatch function."""

    best = {"aep": 0.0, "code": None, "iter": 0}
    attempt_count = [0]
    # Log every attempt: training AEP + silent ROWP AEP (for progress plot)
    attempt_log = []

    def _save_log():
        with open(out_dir / "attempt_log.json", "w") as f:
            json.dump(attempt_log, f, indent=2)

    def _dispatch(name: str, args: dict) -> str:
        """Execute a tool call, return result as string."""

        if name == "read_file":
            path = args["path"]
            # Restrict to playground/pixwake/src only (source code)
            resolved = (playground / path).resolve()
            playground_abs = str(playground.resolve())
            if not str(resolved).startswith(playground_abs):
                return "Error: path must be inside playground/"
            # Block sensitive paths
            rel = str(resolved)
            if any(s in rel for s in [".git/", ".env", "credentials",
                                       ".ssh", ".gnupg", "__pycache__"]):
                return "Error: access denied to that path"
            # Only allow reading source code, not arbitrary files
            if not resolved.exists():
                return f"Error: {path} not found"
            if resolved.is_dir():
                return "Error: use list_files for directories"
            # Block binary files
            suffix = resolved.suffix.lower()
            if suffix in {".pyc", ".so", ".dylib", ".h5", ".nc", ".pkl",
                          ".pickle", ".bin", ".dat", ".npy", ".npz"}:
                return f"Error: cannot read binary file ({suffix})"
            try:
                text = resolved.read_text()
            except UnicodeDecodeError:
                return "Error: cannot read binary file"
            if len(text) > 8000:
                text = text[:8000] + "\n... (truncated)"
            return text

        elif name == "list_files":
            path = args.get("path", ".")
            resolved = (playground / path).resolve()
            playground_abs = str(playground.resolve())
            if not str(resolved).startswith(playground_abs):
                return "Error: path must be inside playground/"
            if any(s in str(resolved) for s in [".git", ".env", ".ssh"]):
                return "Error: access denied to that path"
            if not resolved.exists():
                return f"Error: {path} not found"
            entries = sorted(resolved.iterdir())
            lines = []
            for e in entries[:100]:
                # Skip hidden/sensitive dirs
                if e.name.startswith(".") and e.name not in {".py"}:
                    continue
                rel = e.relative_to(playground.resolve())
                suffix = "/" if e.is_dir() else f" ({e.stat().st_size} bytes)"
                lines.append(f"  {rel}{suffix}")
            return "\n".join(lines) if lines else "(empty directory)"

        elif name == "run_optimizer":
            code = args["code"]
            attempt_count[0] += 1

            safe, reason = safety_check(code)
            if not safe:
                attempt_log.append({
                    "attempt": attempt_count[0],
                    "timestamp": time.time(),
                    "error": f"safety: {reason}",
                })
                _save_log()
                return f"REJECTED: {reason}"

            # Save iteration
            iter_path = out_dir / f"iter_{attempt_count[0]:03d}.py"
            iter_path.write_text(code)

            r = run_on_farm(code, train_farm, playground, results_dir,
                           timeout_s, run_id)
            if "error" in r:
                attempt_log.append({
                    "attempt": attempt_count[0],
                    "timestamp": time.time(),
                    "error": r["error"][:500],
                })
                _save_log()
                return f"ERROR: {r['error'][:2000]}"

            sc = score_layout({"x": r["x"], "y": r["y"]},
                              train_farm, benchmark, wind_csv, playground)
            if "error" in sc:
                attempt_log.append({
                    "attempt": attempt_count[0],
                    "timestamp": time.time(),
                    "error": f"score: {sc['error'][:300]}",
                })
                _save_log()
                return f"Run OK but score error: {sc['error'][:500]}"

            aep = sc["aep_gwh"]
            bl = baselines.get(str(train_farm), {}).get("aep_gwh", 0)

            if aep > best["aep"]:
                best["aep"] = aep
                best["code"] = code
                best["iter"] = attempt_count[0]
                (out_dir / "best_optimizer.py").write_text(code)

            # Silently run on ROWP for progress tracking (LLM doesn't see this)
            rowp_result = _score_on_rowp(code, playground, results_dir,
                                         timeout_s, run_id)

            entry = {
                "attempt": attempt_count[0],
                "timestamp": time.time(),
                "train_aep": aep,
                "train_feasible": sc.get("feasible", None),
                "train_time": r["time"],
                "train_baseline": bl,
            }
            if rowp_result and "error" not in rowp_result:
                entry["rowp_aep"] = rowp_result["aep_gwh"]
                entry["rowp_feasible"] = rowp_result["feasible"]
                entry["rowp_time"] = rowp_result["time"]
            elif rowp_result:
                entry["rowp_error"] = rowp_result["error"][:200]
            attempt_log.append(entry)
            _save_log()

            return (f"AEP: {aep:.2f} GWh (baseline: {bl:.2f}, "
                    f"gap: {aep - bl:+.2f})\n"
                    f"Run time: {r['time']:.1f}s\n"
                    f"Feasible: {sc.get('feasible', 'unknown')}\n"
                    f"Best so far: {best['aep']:.2f} GWh (attempt {best['iter']})")

        elif name == "test_generalization":
            code = args["code"]
            safe, reason = safety_check(code)
            if not safe:
                return f"REJECTED: {reason}"

            # Run on the ROWP problem (held-out case) to check that
            # the script generalizes. The LLM sees PASS/FAIL and
            # feasibility details, but NOT the AEP score.
            rowp_problem = results_dir / "problem_rowp.json"
            if not rowp_problem.exists():
                return ("GENERALIZATION TEST SKIPPED: "
                        "no held-out problem file found.")

            r = run_on_farm(code, "rowp", playground, results_dir,
                            timeout_s, run_id)

            if "error" in r:
                err = r["error"][:2000]
                hints = []
                if "KeyError" in err and "turbine" in err:
                    hints.append("Script does not read info['turbine'] from JSON.")
                if "hub_height" in err:
                    hints.append("Script does not read info['hub_height'] from JSON.")
                return (f"GENERALIZATION TEST FAILED — script errored on a "
                        f"different farm (74 turbines, D=198m, different polygon "
                        f"and wind resource):\n{err}\n"
                        f"{chr(10).join(hints)}\n\n"
                        f"Your script must read ALL parameters from the problem "
                        f"JSON. Do not hardcode turbine data, number of turbines, "
                        f"or boundary-specific values.")

            n = len(r["x"])
            issues = []

            # Check feasibility via ProblemBenchmark
            try:
                pixwake_src = str((playground / "pixwake" / "src").resolve())
                if pixwake_src not in sys.path:
                    sys.path.insert(0, pixwake_src)
                bench_dir = str(playground.parent / "benchmarks")
                if bench_dir not in sys.path:
                    sys.path.insert(0, bench_dir)
                from dei_layout import ProblemBenchmark
                bm = ProblemBenchmark(str(rowp_problem))
                expected_n = bm.n_target
                feas = bm.check_feasibility(r["x"], r["y"])
                if n != expected_n:
                    issues.append(
                        f"WRONG TURBINE COUNT: produced {n}, expected {expected_n}")
                if not feas["spacing_ok"]:
                    issues.append(
                        f"SPACING VIOLATION: min_dist={feas['min_turbine_distance_m']:.1f}m "
                        f"(need {feas['min_spacing_m']:.0f}m). Your perturbation "
                        f"or initialization may be too aggressive for this farm.")
                if not feas["boundary_ok"]:
                    issues.append(
                        f"BOUNDARY VIOLATION: penalty={feas['boundary_penalty']:.4f} "
                        f"(need < 1e-3)")
            except Exception as e:
                issues.append(f"Could not check feasibility: {e}")

            if issues:
                return (f"GENERALIZATION TEST FAILED on held-out farm "
                        f"(74 turbines, D=198m, min_spacing=792m):\n"
                        f"{chr(10).join('  - ' + i for i in issues)}\n"
                        f"Run time: {r['time']:.1f}s\n\n"
                        f"Your script must produce FEASIBLE layouts on any farm. "
                        f"Avoid hardcoding perturbation scales as multiples of D "
                        f"that are too large. Scale perturbations relative to "
                        f"min_spacing or the polygon size, not D.")
            return (f"GENERALIZATION TEST PASSED!\n"
                    f"Script produced a feasible layout on a held-out farm "
                    f"with 74 turbines, D=198m, different polygon and wind.\n"
                    f"Produced {n} turbine positions in {r['time']:.1f}s.\n"
                    f"Your script generalizes across problem configurations.")

        elif name == "get_status":
            bl = baselines.get(str(train_farm), {}).get("aep_gwh", 0)
            return (f"Attempts: {attempt_count[0]}\n"
                    f"Best AEP: {best['aep']:.2f} GWh\n"
                    f"Baseline: {bl:.2f} GWh\n"
                    f"Gap: {best['aep'] - bl:+.2f} GWh")

        else:
            return f"Unknown tool: {name}"

    return _dispatch, best, attempt_log


def get_tool_declarations():
    """Gemini function declarations for the agent's tools."""
    from google.genai import types

    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="read_file",
            description=(
                "Read a file from the playground directory. Use to inspect "
                "pixwake source code, configs, or your previous scripts. "
                "Paths are relative to playground/."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={"path": types.Schema(
                    type="STRING",
                    description="Relative path inside playground/, e.g. 'pixwake/src/pixwake/optim/sgd.py'",
                )},
                required=["path"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_files",
            description="List files in a directory inside the playground.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"path": types.Schema(
                    type="STRING",
                    description="Relative directory path (default: root of playground)",
                )},
            ),
        ),
        types.FunctionDeclaration(
            name="run_optimizer",
            description=(
                "Write and run a complete optimizer script on the training farm. "
                "The script must: (1) import jax and set x64, (2) load problem "
                "from os.environ['FUNWAKE_PROBLEM'], (3) optimize the layout, "
                "(4) write {x, y} to os.environ['FUNWAKE_OUTPUT']. "
                "Returns the scored AEP and feasibility."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={"code": types.Schema(
                    type="STRING",
                    description="Complete Python optimizer script",
                )},
                required=["code"],
            ),
        ),
        types.FunctionDeclaration(
            name="test_generalization",
            description=(
                "Test that your optimizer script generalizes to a DIFFERENT "
                "farm. Runs your script on a held-out farm with 74 turbines, "
                "D=198m, different polygon and wind resource. Reports PASS/"
                "FAIL and feasibility details, but NOT the AEP score. "
                "Your script MUST read ALL parameters from the problem JSON "
                "(turbine curves, hub_height, boundary, n_target). "
                "Call this to verify your script doesn't hardcode values or "
                "use perturbation scales that are too aggressive for "
                "different farm sizes."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={"code": types.Schema(
                    type="STRING",
                    description="Complete Python optimizer script to test",
                )},
                required=["code"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_status",
            description="Get current best AEP, attempt count, and gap vs baseline.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
    ])]


# ── System prompt ───────────────────────────────────────────────────────

def build_system_prompt(baselines: dict, train_farm: int,
                        results_dir: Path, playground: Path) -> str:
    bl = baselines.get(str(train_farm), {})
    bl_aep = bl.get("aep_gwh", 0)

    # Load problem info
    problem_path = results_dir / f"problem_farm{train_farm}.json"
    problem_summary = ""
    if problem_path.exists():
        with open(problem_path) as f:
            info = json.load(f)
        bv = info["boundary_vertices"]
        bv_str = ", ".join(f"({v[0]:.1f}, {v[1]:.1f})" for v in bv)
        problem_summary = (
            f"### Training farm {train_farm}\n"
            f"  n_target: {info['n_target']} turbines\n"
            f"  rotor_diameter: {info['rotor_diameter']}m\n"
            f"  min_spacing: {info['min_spacing_m']}m (4×D)\n"
            f"  boundary ({len(bv)} vertices): [{bv_str}]\n"
        )

    return f"""\
You are an optimization researcher with access to a wind farm layout
optimization codebase (pixwake). Your goal: write a Python optimizer
that MAXIMIZES AEP (Annual Energy Production) and beats the baseline.

## Task

Optimize the placement of ~50 turbines (IEA 15 MW, D=240m) inside a
polygon boundary. Constraints: all turbines inside boundary, minimum
spacing 4×D = 960m between any pair.

## Baseline to beat

500 multi-start topfarm_sgd_solve: **{bl_aep:.2f} GWh**
(max_iter=4000, additional_constant_lr_iterations=2000)

{problem_summary}

## Constraints (exact scorer thresholds)

1. **Boundary**: boundary_penalty(x, y, boundary) < 1e-3
2. **Spacing**: min pairwise distance >= min_spacing × 0.99 = 950.4m

Import penalty functions:
  from pixwake.optim.sgd import boundary_penalty, spacing_penalty

## Your optimizer script must

1. `import jax; jax.config.update("jax_enable_x64", True)` at the top
2. Load problem from `os.environ["FUNWAKE_PROBLEM"]`
3. Optimize the layout
4. Write {{"x": [...], "y": [...]}} to `os.environ["FUNWAKE_OUTPUT"]`

## Problem JSON schema

The problem JSON (loaded from FUNWAKE_PROBLEM) has these keys:
- `rotor_diameter`: float (240.0 for training, may differ for other farms)
- `hub_height`: float (150.0 for training, may differ)
- `min_spacing_m`: float (960.0 = 4×D)
- `n_target`: int (number of turbines)
- `boundary_vertices`: list of [x, y] pairs
- `init_x`, `init_y`: lists of initial turbine positions
- `wind_rose.directions_deg`: list of wind direction bins
- `wind_rose.speeds_ms`: list of mean wind speeds per bin
- `wind_rose.weights`: list of frequency weights per bin
- `turbine.power_curve_ws`: list of wind speeds for power curve
- `turbine.power_curve_kw`: list of power values in kW
- `turbine.ct_curve_ws`: list of wind speeds for Ct curve
- `turbine.ct_curve_ct`: list of thrust coefficient values

IMPORTANT: Your script will be evaluated on a DIFFERENT farm with a
different turbine, boundary, and wind resource. You MUST read ALL
parameters (including turbine curves) from the problem JSON. Do NOT
hardcode turbine data.

## Working baseline template

This COMPLETE script works and scores ~5527 GWh. It reads ALL
configuration from the problem JSON so it works on any farm.

```python
import jax
jax.config.update("jax_enable_x64", True)
import os, json
import jax.numpy as jnp
import numpy as np
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve

with open(os.environ["FUNWAKE_PROBLEM"]) as f:
    info = json.load(f)

D = info["rotor_diameter"]
hub_height = info.get("hub_height", 150.0)
t = info["turbine"]
ws_arr = jnp.array(t["power_curve_ws"], dtype=float)
power = jnp.array(t["power_curve_kw"], dtype=float)
ct_ws = jnp.array(t.get("ct_curve_ws", t["power_curve_ws"]), dtype=float)
ct = jnp.array(t["ct_curve_ct"], dtype=float)
turbine = Turbine(rotor_diameter=D, hub_height=hub_height,
                  power_curve=Curve(ws=ws_arr, values=power),
                  ct_curve=Curve(ws=ct_ws, values=ct))
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
    json.dump({{"x": [float(v) for v in opt_x],
               "y": [float(v) for v in opt_y]}}, f)
```

## Key API facts (DO NOT deviate from these)

- `WakeSimulation(turbine, deficit)` — exactly 2 positional args
- `topfarm_sgd_solve(objective, init_x, init_y, boundary, min_spacing, settings)` — returns `(opt_x, opt_y)` (2 values, NOT 3)
- `SGDSettings(learning_rate=, max_iter=, additional_constant_lr_iterations=, tol=, ks_rho=, spacing_weight=, boundary_weight=, gamma_min_factor=, beta1=, beta2=)`
- `objective(x, y)` must return a scalar (negative AEP)
- `from pixwake.optim.sgd import boundary_penalty, spacing_penalty`

## Tools

You have these tools:
- `read_file(path)` — read pixwake source files to understand the API
- `list_files(path)` — explore the codebase
- `test_generalization(code)` — test your script on a DIFFERENT farm
  with a different turbine, boundary, and wind rose. Call this FIRST
  to make sure your script reads everything from the problem JSON.
  If this fails, your script hardcodes something it shouldn't.
- `run_optimizer(code)` — run a complete script on the training farm and
  get the AEP score back
- `get_status()` — check your best AEP and gap vs baseline

## Strategy

1. Start from the working template above
2. Call `test_generalization` to verify your script reads everything
   from the problem JSON (turbine curves, hub_height, boundary, etc.)
3. Call `run_optimizer` to get the AEP on the training farm
4. Iterate to beat the baseline

Strategy ideas:
- **Multi-start**: run topfarm_sgd_solve from many random initial layouts,
  keep the best. The baseline uses 500 starts — can you do better with
  smarter initialization?
- **Two-stage**: first optimize for feasibility, then AEP
- **Smart initial layouts**: generate grid points inside the polygon,
  filter to inside boundary, use as starting positions
- **Hyperparameter tuning**: learning_rate, max_iter, ks_rho, beta1/beta2
- **Custom optimizer**: write your own gradient-based optimizer using
  jax.grad and the objective/penalty functions directly

CRITICAL: Your script will be evaluated on a HELD-OUT farm with a
DIFFERENT turbine, boundary, and wind resource. Do NOT hardcode any
turbine data, hub height, or boundary — read it all from the JSON.

The script runs in playground/ with pixwake on PYTHONPATH.
"""


# ── Main agent loop ─────────────────────────────────────────────────────

def plot_progress(attempt_log: list, out_dir: Path,
                  train_baseline: float, rowp_baseline: float):
    """Plot training AEP and ROWP AEP vs attempt number."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping plot)")
        return

    successful = [e for e in attempt_log if "train_aep" in e]
    if not successful:
        return

    attempts = [e["attempt"] for e in successful]
    train_aeps = [e["train_aep"] for e in successful]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(attempts, train_aeps, "o-", color="C0", label="Training (DEI farm 1)")
    ax.axhline(train_baseline, color="C0", linestyle="--", alpha=0.5,
               label=f"Train baseline ({train_baseline:.1f})")

    # ROWP points (may have gaps where ROWP errored)
    rowp_attempts = [e["attempt"] for e in successful if "rowp_aep" in e]
    rowp_aeps = [e["rowp_aep"] for e in successful if "rowp_aep" in e]
    if rowp_aeps:
        ax2 = ax.twinx()
        ax2.plot(rowp_attempts, rowp_aeps, "s-", color="C1",
                 label="ROWP (held-out)")
        ax2.axhline(rowp_baseline, color="C1", linestyle="--", alpha=0.5,
                     label=f"ROWP baseline ({rowp_baseline:.1f})")
        ax2.set_ylabel("ROWP AEP (GWh)", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax2.legend(loc="lower right")

    ax.set_xlabel("Attempt")
    ax.set_ylabel("Training AEP (GWh)", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.legend(loc="upper left")
    ax.set_title("Agent Progress: Training vs Held-out AEP")
    fig.tight_layout()
    plot_path = out_dir / "progress.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Progress plot saved to {plot_path}")


def run_agent(provider: str, model: str, api_key: str,
              playground: Path, benchmark: Path, wind_csv: str,
              results_dir: Path, baselines: dict, train_farm: int,
              timeout_s: int, time_budget: float, out_dir: Path,
              run_id: str, temperature: float,
              hot_start: str | None = None):

    out_dir.mkdir(parents=True, exist_ok=True)

    dispatch, best, attempt_log = make_tools(
        playground, results_dir, benchmark, wind_csv, baselines,
        train_farm, timeout_s, run_id, out_dir)

    system = build_system_prompt(baselines, train_farm, results_dir, playground)
    tools = get_tool_declarations()

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=tools,
        temperature=temperature,
        max_output_tokens=8192,
    )

    # Hot start: run a known-good script first and inject the result
    # into the conversation so the model starts from a working baseline
    hot_start_msg = ""
    if hot_start:
        hot_path = Path(hot_start)
        if hot_path.exists():
            hot_code = hot_path.read_text()
            print("Hot start: running seed script...")
            hot_result = dispatch("run_optimizer", {"code": hot_code})
            print(f"  {hot_result[:200]}")
            hot_start_msg = (
                f"\n\nI have already run a seed optimizer for you. Here is the "
                f"result:\n\n{hot_result}\n\n"
                f"Here is the seed script that produced it:\n\n"
                f"```python\n{hot_code}\n```\n\n"
                f"Build on this working script. Focus on strategies to beat "
                f"the baseline — do NOT waste time re-learning the API."
            )
        else:
            print(f"Warning: hot start file {hot_start} not found, skipping")

    # Conversation history
    contents = [types.Content(
        role="user",
        parts=[types.Part(text=(
            "You have a time budget to develop the best wind farm layout "
            "optimizer you can. Your goal is to beat the baseline AEP. "
            "The system prompt contains a working template with correct "
            "API usage — start from that."
            f"{hot_start_msg}\n\n"
            f"You have {time_budget:.0f} seconds. Begin now."
        ))],
    )]

    t0 = time.time()
    turn = 0

    while True:
        elapsed = time.time() - t0
        remaining = time_budget - elapsed

        if remaining <= 0:
            print(f"\n[TIME'S UP] {elapsed:.0f}s elapsed")
            break

        turn += 1
        print(f"\n── Turn {turn} ({elapsed:.0f}s / {time_budget:.0f}s) "
              f"──────────────────────────────")

        try:
            resp = client.models.generate_content(
                model=model, contents=contents, config=config)
        except Exception as e:
            print(f"  LLM error: {e}")
            time.sleep(2)
            continue

        if not resp.candidates:
            print("  No candidates returned")
            break

        candidate = resp.candidates[0]
        model_content = candidate.content

        if model_content is None or not model_content.parts:
            print("  (empty response)")
            if time.time() - t0 < time_budget - 30:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text="Continue. Try a different approach.")],
                ))
                continue
            else:
                break

        # Add model response to history
        contents.append(model_content)

        # Process each part
        has_function_calls = False
        function_results = []

        for part in model_content.parts:
            if hasattr(part, "text") and part.text:
                print(f"  LLM: {part.text[:500]}")

            if hasattr(part, "function_call") and part.function_call:
                has_function_calls = True
                fc = part.function_call
                args = dict(fc.args) if fc.args else {}
                print(f"  → {fc.name}({', '.join(f'{k}={repr(v)[:60]}' for k, v in args.items())})")

                # Execute tool
                tool_t0 = time.time()
                result_str = dispatch(fc.name, args)
                tool_elapsed = time.time() - tool_t0

                # Truncate for display
                display = result_str[:300]
                if len(result_str) > 300:
                    display += "..."
                print(f"    ← ({tool_elapsed:.1f}s) {display}")

                function_results.append(types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result_str},
                    )
                ))

        # If there were function calls, send results back
        if has_function_calls:
            contents.append(types.Content(
                role="user",
                parts=function_results,
            ))

        # Check if model stopped without function calls (thinking/done)
        if not has_function_calls:
            # Check finish reason
            finish = candidate.finish_reason if hasattr(candidate, "finish_reason") else None
            if finish and str(finish) == "STOP":
                # Model chose to stop — nudge it to keep going if time remains
                if remaining > 30:
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part(text=(
                            f"You have {remaining:.0f}s remaining. "
                            f"Keep iterating — try a different strategy."
                        ))],
                    ))
                else:
                    break

    # ── Results ──
    print(f"\n{'='*60}")
    print(f"PROTOTYPING COMPLETE")
    print(f"{'='*60}")
    print(f"  Time: {time.time() - t0:.0f}s")
    print(f"  Best AEP: {best['aep']:.2f} GWh")
    bl_aep = baselines.get(str(train_farm), {}).get("aep_gwh", 0)
    print(f"  Baseline: {bl_aep:.2f} GWh")
    print(f"  Gap: {best['aep'] - bl_aep:+.2f} GWh")

    # ── Held-out evaluation ──
    best_code = best["code"]
    if not best_code and (out_dir / "best_optimizer.py").exists():
        best_code = (out_dir / "best_optimizer.py").read_text()

    # ROWP is the held-out test case (different turbine, polygon, wind)
    rowp_problem = results_dir / "problem_rowp.json"
    rowp_baseline = results_dir / "baseline_rowp.json"

    if best_code:
        print(f"\n{'='*60}")
        print("HELD-OUT EVALUATION: ROWP (74 turbines, IEA 10MW)")
        print(f"{'='*60}")

        if not rowp_problem.exists():
            print("  ROWP problem file not found — skipping")
        else:
            # Run the LLM's optimizer on the ROWP problem
            # The script should read turbine/boundary/wind from the JSON
            print("  Running optimizer on ROWP...")
            r = run_on_farm(best_code, "rowp", playground, results_dir,
                            timeout_s, run_id)
            if "error" in r:
                print(f"  RUN ERROR: {str(r['error'])[:500]}")
                print("  (The optimizer likely hardcodes turbine data instead")
                print("   of reading it from the problem JSON.)")
            else:
                # Score via ProblemBenchmark
                pixwake_src = str((playground / "pixwake" / "src").resolve())
                if pixwake_src not in sys.path:
                    sys.path.insert(0, pixwake_src)
                sys.path.insert(0, str(benchmark.parent))
                from dei_layout import ProblemBenchmark
                bm = ProblemBenchmark(str(rowp_problem))
                aep = bm.score(r["x"], r["y"])
                feas = bm.check_feasibility(r["x"], r["y"])
                feasible = feas["spacing_ok"] and feas["boundary_ok"]

                rowp_bl = 0
                if rowp_baseline.exists():
                    with open(rowp_baseline) as f:
                        rowp_bl = json.load(f).get("aep_gwh", 0)

                print(f"  AEP:      {aep:.2f} GWh")
                print(f"  Baseline: {rowp_bl:.2f} GWh")
                print(f"  Gap:      {aep - rowp_bl:+.2f} GWh")
                print(f"  Feasible: {feasible}")
                if not feasible:
                    print(f"  Details:  spacing_ok={feas['spacing_ok']} "
                          f"(min_dist={feas['min_turbine_distance_m']:.1f}m), "
                          f"boundary_ok={feas['boundary_ok']}")

                history_extra = {
                    "rowp_aep": aep, "rowp_baseline": rowp_bl,
                    "rowp_feasible": feasible, "rowp_time": r["time"],
                }

    # Save history
    history = {
        "time_budget": time_budget,
        "elapsed": time.time() - t0,
        "turns": turn,
        "best_aep": best["aep"],
        "best_iter": best["iter"],
        "baseline_aep": bl_aep,
        "model": model,
        "provider": provider,
    }
    try:
        history.update(history_extra)
    except NameError:
        pass  # ROWP eval didn't run or errored
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved to {out_dir / 'history.json'}")

    # Plot progress
    rowp_bl = 0
    if (results_dir / "baseline_rowp.json").exists():
        with open(results_dir / "baseline_rowp.json") as f:
            rowp_bl = json.load(f).get("aep_gwh", 0)
    plot_progress(attempt_log, out_dir, bl_aep, rowp_bl)


def main():
    p = argparse.ArgumentParser(
        description="Agentic layout optimizer (Claude Code-style)")
    p.add_argument("--playground-dir", default="playground")
    p.add_argument("--benchmark-script", default="benchmarks/dei_layout.py")
    p.add_argument("--wind-csv", required=True)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--provider", default="gemini",
                   choices=["gemini"])
    p.add_argument("--train-farm", type=int, default=1)
    p.add_argument("--timeout", type=int, default=300,
                   help="Per-farm script timeout in seconds")
    p.add_argument("--time-budget", type=float, default=600,
                   help="Total prototyping time budget in seconds (default: 600)")
    p.add_argument("--output-dir", default="results_agent")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--hot-start", default=None,
                   help="Path to a working optimizer script to seed the agent")
    p.add_argument("--run-id", default="agent")
    args = p.parse_args()

    key_files = {"gemini": "~/.gem"}
    key_path = os.path.expanduser(key_files[args.provider])
    api_key = open(key_path).read().strip()

    with open(Path(args.results_dir) / "baselines.json") as f:
        baselines = json.load(f)

    run_agent(
        provider=args.provider,
        model=args.model,
        api_key=api_key,
        playground=Path(args.playground_dir),
        benchmark=Path(args.benchmark_script),
        wind_csv=args.wind_csv,
        results_dir=Path(args.results_dir),
        baselines=baselines,
        train_farm=args.train_farm,
        timeout_s=args.timeout,
        time_budget=args.time_budget,
        out_dir=Path(args.output_dir),
        run_id=args.run_id,
        temperature=args.temperature,
        hot_start=args.hot_start,
    )


if __name__ == "__main__":
    main()
