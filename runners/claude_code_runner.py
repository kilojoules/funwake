"""Claude Code backend — delegates to `claude -p` with --allowedTools.

Architecture:
  Outer loop (this file):  manages time budget, logging, memory context
  Inner loop (Claude Code): reads code, writes optimizers, runs tools

Each iteration = one `claude -p` invocation. Between iterations, this
runner updates the memory file and re-invokes with fresh context.
Alternatively, set iterations=1 and max_turns high for a single session.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from .base import BaseRunner, RunConfig, AttemptResult


class ClaudeCodeRunner(BaseRunner):
    """Claude Code agent via `claude -p` invocations."""

    def __init__(self, config: RunConfig,
                 max_turns_per_iter: int = 30,
                 iterations: int = 0,       # 0 = auto (fill time budget)
                 permission_mode: str = "allowedTools",
                 schedule_only: bool = False):
        super().__init__(config)
        self.max_turns_per_iter = max_turns_per_iter
        self.iterations = iterations
        self.permission_mode = permission_mode
        self.schedule_only = schedule_only

        # Verify claude CLI is available
        result = subprocess.run(["which", "claude"], capture_output=True, text=True)
        if result.returncode != 0:
            raise EnvironmentError("claude CLI not found. Install: npm install -g @anthropic-ai/claude-code")

    def _build_allowed_tools(self) -> list[str]:
        """Build --allowedTools list scoping what Claude Code can do.

        Note: Claude Code's path-scoped Write(dir/*) doesn't work
        reliably. We use broad Read/Write and rely on CLAUDE.md
        instructions + tool script sandboxing for safety.
        """
        sched_flag = " --schedule-only" if self.schedule_only else ""
        return [
            "Read",
            "Write",
            # ONLY these tool scripts — no other Bash commands.
            f"Bash(python tools/run_tests.py *)",
            f"Bash(python tools/run_optimizer.py *{sched_flag})",
            f"Bash(python tools/test_generalization.py *{sched_flag})",
            "Bash(python tools/get_status.py *)",
            "Grep",
            "Glob",
        ]

    def _build_claude_md(self) -> str:
        """Generate .claude/CLAUDE.md for this session."""
        if self.schedule_only:
            return self._build_schedule_claude_md()
        return f"""\
# FunWake Optimizer Agent

You are optimizing wind farm layouts. Write `optimize()` functions that
maximize AEP (Annual Energy Production).

## Rules
- Write optimizers to `{self.config.output_dir}/iter_NNN.py`
- You MUST score every optimizer by running: `python tools/run_optimizer.py <script>`
- Run tests first: `python tools/run_tests.py <script> --quick`
- Test generalization: `python tools/test_generalization.py <script>`
- Check status: `python tools/get_status.py --log {self.log_path}`
- You CANNOT modify files in `playground/pixwake/`, `benchmarks/`, or `tools/`
- You CANNOT run Python directly — only through the tool scripts above
- Baseline to beat: {self._get_baseline_aep():.1f} GWh

## CRITICAL: Timeout constraint
- Each optimizer run times out at {self.config.timeout_per_run}s
- Your optimizer MUST complete within this budget
- Use 1-3 multi-starts, NOT 10+
- The seed optimizer (single start) runs in ~23s — that's your reference

## Function signature (MUST match exactly)
```python
def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    # Returns (opt_x, opt_y) — JAX arrays
```

## Key pixwake API
- `SGDSettings(learning_rate, max_iter, additional_constant_lr_iterations, tol, ...)`
- `topfarm_sgd_solve(objective, init_x, init_y, boundary, min_spacing, settings)`
- `sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None).power()` → shape (n_findex, n_turbines)

## Workflow
1. Read `{self.config.output_dir}/agent_memory.md` for current status and history
2. Read pixwake source if needed (`playground/pixwake/src/pixwake/`)
3. Write an optimizer script
4. Run tests → fix if failing
5. Score → compare to baseline and previous best
6. Test generalization → check PASS/FAIL
7. Iterate: try different strategies (initialization, hyperparams, custom optimizers)

## Environment
PYTHONPATH includes `playground/pixwake/src`. JAX_ENABLE_X64=True is set.
"""

    def _build_schedule_claude_md(self) -> str:
        """CLAUDE.md for schedule-only mode."""
        return f"""\
# FunWake Schedule Designer

You are designing the **learning rate and penalty schedule** for a wind
farm layout optimizer. A fixed skeleton handles initialization, gradient
computation, and Adam updates. You control ONLY how four parameters
change over the course of optimization.

## Your task

Write a `schedule_fn` that returns (lr, alpha, beta1, beta2) at each step:

```python
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    \"\"\"Control the optimizer schedule.

    Args:
        step: current iteration (0 to total_steps-1)
        total_steps: total iterations (8000)
        lr0: initial learning rate (50.0)
        alpha0: initial penalty weight (from gradient magnitude)

    Returns:
        lr: learning rate for this step
        alpha: constraint penalty multiplier (higher = stricter constraints)
        beta1: Adam first moment decay (0 to 1)
        beta2: Adam second moment decay (0 to 1)
    \"\"\"
    ...
    return lr, alpha, beta1, beta2
```

## What the skeleton does (you CANNOT change this)
- Wind-direction-aware grid initialization inside the polygon
- Computes gradients of AEP objective and constraint penalties (boundary + spacing)
- Combines gradients: `grad = grad_obj + alpha * grad_constraint`
- Adam update with your beta1, beta2 at your learning rate
- Runs for 8000 steps total

## What you control
- **lr**: learning rate — how big each step is
- **alpha**: penalty weight — how much to prioritize feasibility vs AEP
- **beta1**: Adam first moment decay — momentum (0.9 = high momentum, 0.1 = low)
- **beta2**: Adam second moment decay — adaptive scaling (0.999 = standard)

## Key insight
The baseline solver couples alpha to 1/lr: as lr decays, alpha increases.
This ensures feasibility in late iterations. Can you do better?

## Rules
- Write schedule files to `{self.config.output_dir}/iter_NNN.py`
- Each file must define ONLY `schedule_fn(step, total_steps, lr0, alpha0)`
- Do NOT write `optimize()` — it will be rejected
- Do NOT import topfarm_sgd_solve — you are replacing it
- Score: `python tools/run_optimizer.py <script> --schedule-only`
- Test: `python tools/run_tests.py <script> --quick`
- Generalize: `python tools/test_generalization.py <script> --schedule-only`
- Status: `python tools/get_status.py --log {self.log_path}`
- Baseline: {self._get_baseline_aep():.1f} GWh
- Timeout: {self.config.timeout_per_run}s per run

## Ideas to try
- Cosine annealing of lr with warm restarts
- Cyclic alpha (high → low → high) instead of monotonic
- Phase transitions: different beta1/beta2 in early vs late iterations
- Exponential vs polynomial lr decay
- Alpha proportional to gradient magnitude (adaptive)
- Standard Adam (beta1=0.9, beta2=0.999) vs TopFarm (0.1, 0.2)

## Workflow
1. Read `{self.config.output_dir}/agent_memory.md` for status
2. Read `results/seed_schedule.py` to see the starting schedule
3. Read `playground/skeleton.py` to understand the fixed optimizer
4. Write a new schedule, score it, iterate
"""

    def _write_memory_file(self):
        """Write agent_memory.md to the run-scoped output dir."""
        from .memory import render_agent_memory
        content = render_agent_memory(self.session, self.history, self.attempts)
        Path(self.memory_path).write_text(content)

    def _setup_claude_config(self):
        """Set up .claude/ directory with CLAUDE.md."""
        os.makedirs(".claude", exist_ok=True)
        Path(".claude/CLAUDE.md").write_text(self._build_claude_md())

    def _invoke_claude(self, prompt: str) -> str:
        """Run a single `claude -p` invocation and return stdout."""
        allowed = self._build_allowed_tools()

        cmd = [
            "claude", "-p", prompt,
            "--max-turns", str(self.max_turns_per_iter),
            "--output-format", "text",
        ]

        # Add allowed tools
        for tool in allowed:
            cmd.extend(["--allowedTools", tool])

        env = {
            **os.environ,
            "PYTHONPATH": self.config.pythonpath,
            "JAX_ENABLE_X64": "True",
        }

        # Timeout: generous but bounded
        timeout = min(self.max_turns_per_iter * 120, int(self.time_remaining()) + 60)

        print(f"[claude -p] Invoking with {self.max_turns_per_iter} max turns, "
              f"{self.time_remaining()/60:.0f} min remaining...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=timeout, env=env,
                cwd=os.getcwd()  # run from project root
            )
            if result.returncode != 0:
                print(f"[claude -p] Non-zero exit: {result.returncode}")
                if result.stderr:
                    print(f"[claude -p] stderr: {result.stderr[:500]}")
            return result.stdout
        except subprocess.TimeoutExpired:
            print(f"[claude -p] Timed out after {timeout}s")
            return ""

    def _parse_new_attempts(self) -> list[dict]:
        """Check attempt_log.json for new entries added by tool scripts."""
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path) as f:
            current = json.load(f)
        new_count = len(current) - len(self.attempts)
        if new_count > 0:
            return current[-new_count:]
        return []

    def run(self):
        """Main loop: repeated `claude -p` invocations with memory updates."""
        self.start_time = time.time()
        self.session.start_time = self.start_time
        self._setup_claude_config()

        # Determine iteration count
        if self.iterations > 0:
            n_iters = self.iterations
        else:
            # Estimate: ~3-5 min per iteration, fill the time budget
            n_iters = max(1, self.config.time_budget // 180)

        # Hot-start context
        seed_context = ""
        if self.config.hot_start and os.path.exists(self.config.hot_start):
            seed = Path(self.config.hot_start).read_text()
            seed_context = f"\n\nHere's a seed optimizer to start from:\n```python\n{seed}\n```"

        for i in range(n_iters):
            if self.time_remaining() <= 30:
                print(f"[iter {i+1}] Time's up.")
                break

            # Update memory file before each invocation
            self._write_memory_file()

            # Build the prompt for this iteration
            mem_path = self.memory_path  # run-scoped
            if i == 0:
                prompt = (
                    f"Read {mem_path} for context. Explore the pixwake codebase "
                    f"in playground/pixwake/src/, understand the API, then write and "
                    f"test an optimizer. Score it and test generalization.{seed_context}"
                )
            else:
                prompt = (
                    f"Read {mem_path} for your updated status and history. "
                    f"You have {self.time_remaining()/60:.0f} minutes remaining. "
                    f"Write an improved optimizer based on what you've learned. "
                    f"Try a different strategy than your last attempt. "
                    f"Test it, score it, and check generalization."
                )

            output = self._invoke_claude(prompt)

            if output:
                print(f"[iter {i+1}] Claude output ({len(output)} chars):")
                # Print last few lines as summary
                lines = output.strip().split("\n")
                for line in lines[-10:]:
                    print(f"  {line}")

            # Sync attempt log — Claude Code's tool runs may have updated it
            # via the tool scripts writing to attempt_log.json
            self._sync_attempts()

            print(f"[iter {i+1}] Attempts: {len(self.attempts)}, "
                  f"Best: {self.best_aep:.1f} GWh, "
                  f"Elapsed: {self.elapsed_minutes():.1f} min\n")

        print(f"\nDone. Best AEP: {self.best_aep:.1f} GWh over {len(self.attempts)} attempts.")

    def _sync_attempts(self):
        """Re-read the attempt log in case tool scripts updated it."""
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                self.attempts = json.load(f)
            successes = [a for a in self.attempts if "train_aep" in a]
            if successes:
                best = max(successes, key=lambda a: a["train_aep"])
                self.best_aep = best["train_aep"]
