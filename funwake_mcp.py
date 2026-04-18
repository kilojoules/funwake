#!/usr/bin/env python
"""FunWake MCP Server

Exposes the FunWake tool scripts as proper MCP tools over stdio so any
MCP-capable client (Claude Code, Claude Desktop, Cursor …) can call them
without shell-outs or subprocess wrangling.

Tools:
  run_optimizer     — score a schedule / optimizer script on a farm
  run_tests         — run the unit-test suite against a script
  get_status        — current best AEP vs baseline + attempt summary
  read_file         — sandbox-safe file reads (playground/ and results/ only)
  write_file        — save a script to the output dir after sandbox check

Usage:
  # Minimal (defaults to playground/problem.json + results/baselines.json)
  python funwake_mcp.py

  # Full config for a schedule-only run
  python funwake_mcp.py \\
      --output-dir results_agent_claude_sched_s1 \\
      --problem    results/problem_farm1.json \\
      --baselines  results/baselines.json \\
      --train-farm 1 \\
      --schedule-only \\
      --timeout    60

Add to claude_desktop_config.json:
  {
    "mcpServers": {
      "funwake": {
        "command": "python",
        "args": ["/path/to/funwake_mcp.py",
                 "--output-dir", "results_agent_mcp_s1",
                 "--schedule-only"],
        "env": {
          "JAX_ENABLE_X64": "True",
          "PYTHONPATH": "playground/pixwake/src"
        }
      }
    }
  }
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# ── Parse server config from CLI ──────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FunWake MCP server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir",   default="results_agent_mcp",
                   help="Directory for iter_NNN.py scripts and attempt_log.json")
    p.add_argument("--problem",      default="playground/problem.json",
                   help="Training-farm problem JSON")
    p.add_argument("--baselines",    default="results/baselines.json",
                   help="Baselines JSON (keyed by farm id)")
    p.add_argument("--train-farm",   default="1",
                   help="Key inside baselines.json for the training baseline")
    p.add_argument("--timeout",      type=int, default=180,
                   help="Default per-script timeout (seconds)")
    p.add_argument("--schedule-only", action="store_true",
                   help="Require schedule_fn() interface, reject optimize()")
    p.add_argument("--project-root", default=None,
                   help="Project root (default: directory containing this file)")
    return p.parse_args()


# Resolve paths relative to the project root so the server can be launched
# from any working directory.
_ARGS = _parse_args()
_PROJECT_ROOT = Path(
    _ARGS.project_root if _ARGS.project_root else Path(__file__).parent
).resolve()
_TOOLS_DIR   = _PROJECT_ROOT / "tools"
_PIXWAKE_SRC = _PROJECT_ROOT / "playground" / "pixwake" / "src"

_OUTPUT_DIR  = Path(_ARGS.output_dir)
if not _OUTPUT_DIR.is_absolute():
    _OUTPUT_DIR = _PROJECT_ROOT / _OUTPUT_DIR
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_LOG_PATH    = _OUTPUT_DIR / "attempt_log.json"
_MEMORY_PATH = _OUTPUT_DIR / "agent_memory.md"
_PROBLEM     = _ARGS.problem
_BASELINES   = _ARGS.baselines
_TRAIN_FARM  = _ARGS.train_farm
_TIMEOUT     = _ARGS.timeout
_SCHED_ONLY  = _ARGS.schedule_only
_START_TIME  = time.time()

# ── Subprocess environment ─────────────────────────────────────────────────────

def _env() -> dict[str, str]:
    """Environment for child processes: inherits PATH, adds PYTHONPATH."""
    e = {
        "PATH":          os.environ.get("PATH", ""),
        "HOME":          os.environ.get("HOME", ""),
        "TMPDIR":        os.environ.get("TMPDIR", "/tmp"),
        "JAX_ENABLE_X64": "True",
        "PYTHONPATH":    f"{_PIXWAKE_SRC}:{os.environ.get('PYTHONPATH', '')}",
    }
    # Forward GPU / HPC visibility vars (LUMI, CUDA, ROCm)
    for k, v in os.environ.items():
        if k.startswith(("ROCR_", "HIP_", "CUDA_", "HSA_", "NCCL_")):
            e[k] = v
    return e


def _run(cmd: list[str], timeout: int) -> tuple[str, str, int]:
    """Run a subprocess, return (stdout, stderr, returncode)."""
    try:
        r = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout,
            env=_env(),
            cwd=str(_PROJECT_ROOT),
        )
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "", f"Process timed out after {timeout}s", -1
    except Exception as exc:
        return "", str(exc), -1


# ── Sandbox helper (reuses sandbox.py from the project) ───────────────────────

def _sandbox_check(code: str) -> tuple[bool, str]:
    """Import and run check_code_safety from the project's sandbox module."""
    sandbox_path = _PROJECT_ROOT / "sandbox.py"
    if not sandbox_path.exists():
        return True, ""          # no sandbox module — pass through
    # Add project root to sys.path temporarily
    sys.path.insert(0, str(_PROJECT_ROOT))
    try:
        from sandbox import check_code_safety  # type: ignore
        return check_code_safety(code)
    except ImportError:
        return True, ""
    finally:
        if str(_PROJECT_ROOT) in sys.path:
            sys.path.remove(str(_PROJECT_ROOT))


# ── Agent memory refresh ──────────────────────────────────────────────────────

def _get_baseline_aep() -> float:
    """Read baseline AEP from baselines JSON."""
    try:
        bl = json.loads((_PROJECT_ROOT / _BASELINES).read_text())
        return bl.get(_TRAIN_FARM, {}).get("aep_gwh", 0.0)
    except (json.JSONDecodeError, OSError, KeyError):
        return 0.0


def _get_time_budget() -> float:
    """Infer time budget from CLI args or default to 5h."""
    return float(os.environ.get("FUNWAKE_TIME_BUDGET", 18000))


def _refresh_agent_memory():
    """Regenerate agent_memory.md, preserving agent-authored notes."""
    try:
        sys.path.insert(0, str(_PROJECT_ROOT / "runners"))
        from memory_template import refresh_memory
        attempts = []
        if _LOG_PATH.exists():
            attempts = json.loads(_LOG_PATH.read_text())
        refresh_memory(
            memory_path=str(_MEMORY_PATH),
            attempt_log=attempts,
            baseline_aep=_get_baseline_aep(),
            time_budget_s=_get_time_budget(),
            elapsed_s=time.time() - _START_TIME,
        )
    except Exception:
        pass  # memory refresh is best-effort


# ── Attempt-count helper ───────────────────────────────────────────────────────

def _next_attempt_number() -> int:
    if not _LOG_PATH.exists():
        return 1
    try:
        return len(json.loads(_LOG_PATH.read_text())) + 1
    except (json.JSONDecodeError, OSError):
        return 1


# ── MCP server ─────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "funwake",
    instructions=(
        "FunWake optimizer toolkit. "
        "Use write_file to save a script, then run_optimizer to score it, "
        "run_tests to check correctness, and get_status to track progress."
    ),
)


# ─── Tool: run_optimizer ───────────────────────────────────────────────────────

@mcp.tool()
def run_optimizer(
    script_path: str,
    problem: str | None = None,
    timeout: int | None = None,
    schedule_only: bool | None = None,
) -> dict[str, Any]:
    """Score an optimizer or schedule script on a farm.

    Delegates to tools/run_optimizer.py and appends the result to
    attempt_log.json.  Returns a dict with these keys on success:

        aep_gwh   float   Annual Energy Production in GWh
        feasible  bool    Whether the layout satisfies all constraints
        time_s    float   Wall-clock seconds for this run
        baseline  float   Baseline AEP for comparison
        gap       float   aep_gwh − baseline

    On error, returns {"error": "<message>"}.

    Args:
        script_path:   Path to the Python script to evaluate.
                       Relative paths are resolved from the project root.
        problem:       Override the training-farm problem JSON.
                       Defaults to the server's --problem setting.
        timeout:       Per-run timeout in seconds.
                       Defaults to the server's --timeout setting.
        schedule_only: If True, require schedule_fn() and reject optimize().
                       Defaults to the server's --schedule-only flag.
    """
    resolved = (
        Path(script_path) if Path(script_path).is_absolute()
        else _PROJECT_ROOT / script_path
    )
    if not resolved.exists():
        return {"error": f"Script not found: {script_path}"}

    prob    = problem       if problem       is not None else _PROBLEM
    tmo     = timeout       if timeout       is not None else _TIMEOUT
    s_only  = schedule_only if schedule_only is not None else _SCHED_ONLY

    cmd = [
        sys.executable,
        str(_TOOLS_DIR / "run_optimizer.py"),
        str(resolved),
        "--problem",    str(_PROJECT_ROOT / prob),
        "--baselines",  str(_PROJECT_ROOT / _BASELINES),
        "--train-farm", _TRAIN_FARM,
        "--timeout",    str(tmo),
        "--log",        str(_LOG_PATH),
    ]
    if s_only:
        cmd.append("--schedule-only")

    stdout, stderr, rc = _run(cmd, timeout=tmo + 60)

    # Refresh agent_memory.md after each eval
    _refresh_agent_memory()

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"error": stderr or stdout or f"exit code {rc}"}


# ─── Tool: run_tests ──────────────────────────────────────────────────────────

@mcp.tool()
def run_tests(
    script_path: str,
    quick: bool = True,
) -> dict[str, Any]:
    """Run the unit-test suite (tools/run_tests.py) against a script.

    Always run this before run_optimizer to catch signature errors,
    import problems, and basic correctness issues cheaply.

    Returns:
        passed  bool    True if all tests passed
        output  str     Test output (stdout)
        errors  str     Stderr / traceback if any

    Args:
        script_path: Path to the Python script to test.
        quick:       If True, pass --quick to skip the full-farm test
                     (faster; still catches signature/import errors).
    """
    resolved = (
        Path(script_path) if Path(script_path).is_absolute()
        else _PROJECT_ROOT / script_path
    )
    if not resolved.exists():
        return {"passed": False, "output": "", "errors": f"Script not found: {script_path}"}

    cmd = [
        sys.executable,
        str(_TOOLS_DIR / "run_tests.py"),
        str(resolved),
    ]
    if quick:
        cmd.append("--quick")

    stdout, stderr, rc = _run(cmd, timeout=180)

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "passed": rc == 0,
            "output": stdout[:4000],
            "errors": stderr[:2000] if stderr else None,
        }


# ─── Tool: get_status ─────────────────────────────────────────────────────────

@mcp.tool()
def get_status(
    train_farm: str | None = None,
) -> dict[str, Any]:
    """Return a summary of the current agent session.

    Reads attempt_log.json from the server's output directory and the
    baselines file to compute progress metrics.

    Returns:
        attempts   int    Total attempts logged
        successes  int    Attempts that returned an AEP
        errors     int    Attempts that returned an error
        best_aep   float  Best feasible training AEP seen so far
        baseline   float  Baseline AEP for the training farm
        gap        float  best_aep − baseline
        log_path   str    Path to attempt_log.json

    Args:
        train_farm: Override the farm key used to look up the baseline.
                    Defaults to the server's --train-farm setting.
    """
    farm = train_farm if train_farm is not None else _TRAIN_FARM

    cmd = [
        sys.executable,
        str(_TOOLS_DIR / "get_status.py"),
        "--log",        str(_LOG_PATH),
        "--baselines",  str(_PROJECT_ROOT / _BASELINES),
        "--train-farm", farm,
    ]
    stdout, stderr, rc = _run(cmd, timeout=15)

    try:
        data = json.loads(stdout)
        data["log_path"] = str(_LOG_PATH)
        return data
    except json.JSONDecodeError:
        # Fall back to direct computation so the tool never fails silently
        attempts: list[dict] = []
        if _LOG_PATH.exists():
            try:
                attempts = json.loads(_LOG_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        baseline = 0.0
        try:
            bl_data = json.loads((_PROJECT_ROOT / _BASELINES).read_text())
            baseline = bl_data.get(farm, {}).get("aep_gwh", 0.0)
        except (json.JSONDecodeError, OSError, KeyError):
            pass

        successes = [a for a in attempts if "train_aep" in a]
        errors    = [a for a in attempts if "error"     in a]
        best_aep  = max((a["train_aep"] for a in successes), default=0.0)

        return {
            "attempts":  len(attempts),
            "successes": len(successes),
            "errors":    len(errors),
            "best_aep":  round(best_aep, 2),
            "baseline":  round(baseline, 2),
            "gap":       round(best_aep - baseline, 2),
            "log_path":  str(_LOG_PATH),
        }


# ─── Tool: read_file ──────────────────────────────────────────────────────────

_READ_ALLOWLIST = (
    "playground/",
    "results/",
    "runners/",
    "tools/",
    "sandbox.py",
    "CLAUDE.md",
    "README.md",
)
_MAX_READ_CHARS = 24_000  # ~6 k tokens


@mcp.tool()
def read_file(path: str) -> str:
    """Read a file from the project.

    For safety, only paths inside these directories are readable:
      playground/  results/  runners/  tools/
    plus sandbox.py, CLAUDE.md, and README.md.

    Returns the file's text content, truncated to ~24 000 characters
    with a notice if truncation occurred.

    Args:
        path: Relative path from the project root
              (e.g. "playground/pixwake/src/pixwake/optim/sgd.py").
    """
    # Normalise and guard against traversal
    norm = Path(path).as_posix().lstrip("/")
    if ".." in norm:
        return f"[ERROR] Path traversal not allowed: {path!r}"

    allowed = any(
        norm == entry.rstrip("/") or norm.startswith(entry)
        for entry in _READ_ALLOWLIST
    )
    if not allowed:
        allowed_str = ", ".join(_READ_ALLOWLIST)
        return (
            f"[ERROR] Read denied: {path!r}\n"
            f"Allowed prefixes: {allowed_str}"
        )

    full = _PROJECT_ROOT / norm
    if not full.exists():
        return f"[ERROR] File not found: {path!r}"
    if full.is_dir():
        # Return a directory listing instead of failing
        entries = sorted(full.iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = [f"[DIRECTORY: {path}]"]
        for e in entries:
            indicator = "/" if e.is_dir() else ""
            lines.append(f"  {e.name}{indicator}")
        return "\n".join(lines)

    try:
        text = full.read_text(errors="replace")
    except OSError as exc:
        return f"[ERROR] Could not read {path!r}: {exc}"

    if len(text) > _MAX_READ_CHARS:
        text = text[:_MAX_READ_CHARS]
        text += f"\n\n[... TRUNCATED at {_MAX_READ_CHARS} chars ...]"
    return text


# ─── Tool: write_file ─────────────────────────────────────────────────────────

@mcp.tool()
def write_file(
    filename: str,
    content: str,
    auto_number: bool = True,
) -> dict[str, Any]:
    """Save a script to the output directory after a sandbox safety check.

    The sandbox blocks dangerous imports (os, subprocess, socket …),
    dangerous builtins (exec, eval, open …), and dunder-escape patterns.
    Safe code is saved; blocked code returns an error with the reason.

    If auto_number is True (default) and filename matches iter_NNN.py,
    the NNN is replaced with the next sequential attempt number so
    scripts are always numbered correctly regardless of what the caller
    supplies.

    Returns:
        saved_path  str   Absolute path where the file was written
        attempt     int   Attempt number (from attempt_log.json count)
        sandbox     str   "passed" or reason for failure

    Args:
        filename:    Target filename, e.g. "iter_001.py" or "my_schedule.py".
                     Directory components are stripped for safety.
        content:     Python source code to save.
        auto_number: Replace the iter_NNN number with the correct sequence
                     number (recommended: True).
    """
    # Strip any directory component — writes always go to output_dir
    safe_name = Path(filename).name
    if not safe_name.endswith(".py"):
        safe_name += ".py"

    # Sandbox check
    safe, reason = _sandbox_check(content)
    if not safe:
        return {
            "saved_path": None,
            "attempt":    None,
            "sandbox":    f"BLOCKED: {reason}",
            "error":      reason,
        }

    # Auto-numbering: replace NNN in iter_NNN.py with the correct number
    attempt_num = _next_attempt_number()
    if auto_number and safe_name.startswith("iter_") and safe_name.endswith(".py"):
        safe_name = f"iter_{attempt_num:03d}.py"

    dest = _OUTPUT_DIR / safe_name
    dest.write_text(content)

    return {
        "saved_path": str(dest),
        "attempt":    attempt_num,
        "sandbox":    "passed",
    }


# ─── Tool: read_memory ───────────────────────────────────────────────────────

@mcp.tool()
def read_memory() -> str:
    """Read the agent_memory.md file for this session.

    This is the primary context source. Read it at the start of every
    turn to see: current status, top scripts, your key findings, and
    your planned next experiments.

    Returns the full contents of agent_memory.md, or a default template
    if the file doesn't exist yet.
    """
    if _MEMORY_PATH.exists():
        return _MEMORY_PATH.read_text()
    # Bootstrap with empty template
    _refresh_agent_memory()
    if _MEMORY_PATH.exists():
        return _MEMORY_PATH.read_text()
    return "# Agent Memory\n\nNo data yet. Run an optimizer to populate.\n"


# ─── Tool: update_memory ────────────────────────────────────────────────────

@mcp.tool()
def update_memory(
    key_findings: str | None = None,
    next_experiments: str | None = None,
) -> dict[str, str]:
    """Update the agent-authored sections of agent_memory.md.

    The Status and Top Scripts sections are auto-generated from
    attempt_log.json. You control Key Findings and Next Experiments.

    Call this after each eval to record what you learned and plan
    your next attempt. Your notes persist across memory refreshes.

    Args:
        key_findings:    Markdown for the '## Key Findings' section.
                         Replaces the entire section if provided.
        next_experiments: Markdown for the '## Next Experiments' section.
                         Replaces the entire section if provided.
    """
    existing = ""
    if _MEMORY_PATH.exists():
        existing = _MEMORY_PATH.read_text()

    # Parse existing agent notes
    sys.path.insert(0, str(_PROJECT_ROOT / "runners"))
    from memory_template import extract_agent_notes
    notes = extract_agent_notes(existing)

    if key_findings is not None:
        # Replace Key Findings section
        if "## Key Findings" in notes:
            # Find end of Key Findings (next ## header or end)
            start = notes.index("## Key Findings")
            rest = notes[start + len("## Key Findings"):]
            next_header = rest.find("\n## ")
            if next_header >= 0:
                after = rest[next_header:]
            else:
                after = ""
            notes = f"## Key Findings\n{key_findings}\n{after}"
        else:
            notes = f"## Key Findings\n{key_findings}\n\n{notes}"

    if next_experiments is not None:
        if "## Next Experiments" in notes:
            start = notes.index("## Next Experiments")
            rest = notes[start + len("## Next Experiments"):]
            next_header = rest.find("\n## ")
            if next_header >= 0:
                after = rest[next_header:]
            else:
                after = ""
            notes = notes[:notes.index("## Next Experiments")] + \
                    f"## Next Experiments\n{next_experiments}\n{after}"
        else:
            notes += f"\n## Next Experiments\n{next_experiments}\n"

    # Re-render with updated notes
    attempts = []
    if _LOG_PATH.exists():
        try:
            attempts = json.loads(_LOG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    from memory_template import render_memory
    content = render_memory(
        attempt_log=attempts,
        baseline_aep=_get_baseline_aep(),
        time_budget_s=_get_time_budget(),
        elapsed_s=time.time() - _START_TIME,
        agent_notes=notes,
    )
    _MEMORY_PATH.write_text(content)

    return {"status": "updated", "memory_path": str(_MEMORY_PATH)}


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    # Log startup config to stderr (visible in Claude Code / MCP logs)
    print(
        f"[funwake-mcp] starting\n"
        f"  project_root : {_PROJECT_ROOT}\n"
        f"  output_dir   : {_OUTPUT_DIR}\n"
        f"  problem      : {_PROBLEM}\n"
        f"  baselines    : {_BASELINES}\n"
        f"  train_farm   : {_TRAIN_FARM}\n"
        f"  timeout      : {_TIMEOUT}s\n"
        f"  schedule_only: {_SCHED_ONLY}",
        file=sys.stderr,
        flush=True,
    )

    mcp.run(transport="stdio")
