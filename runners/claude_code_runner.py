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
                 permission_mode: str = "allowedTools"):
        super().__init__(config)
        self.max_turns_per_iter = max_turns_per_iter
        self.iterations = iterations
        self.permission_mode = permission_mode

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
        return [
            "Read",
            "Write",
            "Bash(python tools/run_tests.py *)",
            "Bash(python tools/run_optimizer.py *)",
            "Bash(python tools/test_generalization.py *)",
            "Bash(python tools/get_status.py *)",
            "Bash(cat *)",
            "Bash(ls *)",
            "Bash(mkdir *)",
            "Grep",
            "Glob",
        ]

    def _build_claude_md(self) -> str:
        """Generate .claude/CLAUDE.md for this session."""
        return f"""\
# FunWake Optimizer Agent

You are optimizing wind farm layouts. Write `optimize()` functions that
maximize AEP (Annual Energy Production).

## Rules
- Write optimizers to `{self.config.output_dir}/iter_NNN.py`
- Also copy to `playground/_generated_opt.py` for the harness
- Run tests before scoring: `python tools/run_tests.py <script> {self.config.train_problem}`
- Score on training farm: `python tools/run_optimizer.py <script>`
- Test generalization: `python tools/test_generalization.py <script>`
- Check status: `python tools/get_status.py --log {self.log_path}`
- You CANNOT modify files in `playground/pixwake/`, `benchmarks/`, or `tools/`
- Baseline to beat: {self._get_baseline_aep():.1f} GWh

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
