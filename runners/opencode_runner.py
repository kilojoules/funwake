"""OpenCode backend — delegates to `opencode run` for open-source models.

Architecture mirrors ClaudeCodeRunner:
  Outer loop (this file):  manages time budget, logging, memory context
  Inner loop (OpenCode):   reads code, writes optimizers, runs tools

OpenCode connects to a local vLLM server via OpenAI-compatible API.
Configuration is written to opencode.json before each invocation.
"""
import json
import os
import subprocess
import time
from pathlib import Path

from .base import BaseRunner, RunConfig, AttemptResult


class OpenCodeRunner(BaseRunner):
    """OpenCode agent via `opencode run` invocations."""

    def __init__(self, config: RunConfig,
                 model: str = "vllm/qwen2.5-coder-32b",
                 base_url: str = "http://localhost:8000",
                 max_turns_per_iter: int = 30,
                 iterations: int = 0,
                 schedule_only: bool = False):
        super().__init__(config)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_turns_per_iter = max_turns_per_iter
        self.iterations = iterations
        self.schedule_only = schedule_only

        # Verify opencode CLI is available
        result = subprocess.run(["which", "opencode"], capture_output=True, text=True)
        if result.returncode != 0:
            raise EnvironmentError(
                "opencode CLI not found. Install: curl -fsSL https://opencode.ai/install | bash"
            )

    def _write_opencode_config(self, vllm_model_id: str = None):
        """Write opencode.json pointing at vLLM via OpenAI-compatible env vars.

        Uses the built-in openai provider with OPENAI_BASE_URL redirected
        to the local vLLM server. No npm/Node.js required.
        """
        if vllm_model_id is None:
            vllm_model_id = self.model.split("/", 1)[-1] if "/" in self.model else self.model
        self._vllm_model_id = vllm_model_id

        config = {
            "$schema": "https://opencode.ai/config.json",
            "model": f"openai/{vllm_model_id}",
            "small_model": f"openai/{vllm_model_id}",
            "instructions": self._build_instructions(),
        }

        Path("opencode.json").write_text(json.dumps(config, indent=2))

    def _build_instructions(self) -> str:
        """Build instructions for OpenCode (equivalent to CLAUDE.md)."""
        if self.schedule_only:
            return self._build_schedule_instructions()

        return f"""\
You are optimizing wind farm layouts. Write `optimize()` functions that
maximize AEP (Annual Energy Production).

## Rules
- Write optimizers to `{self.config.output_dir}/iter_NNN.py`
- Score every optimizer: `python tools/run_optimizer.py <script>`
- Run tests first: `python tools/run_tests.py <script> --quick`
- Test generalization: `python tools/test_generalization.py <script>`
- Check status: `python tools/get_status.py --log {self.log_path}`
- Do NOT modify files in `playground/pixwake/`, `benchmarks/`, or `tools/`
- Baseline to beat: {self._get_baseline_aep():.1f} GWh
- Each optimizer times out at {self.config.timeout_per_run}s

## Function signature (MUST match exactly)
```python
def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    # Returns (opt_x, opt_y) — JAX arrays
```

## Workflow
1. Read `{self.config.output_dir}/agent_memory.md` for status
2. Read pixwake source if needed (`playground/pixwake/src/pixwake/`)
3. Write an optimizer, test it, score it, check generalization
4. Iterate with different strategies
"""

    def _build_schedule_instructions(self) -> str:
        """Instructions for schedule-only mode."""
        return f"""\
You are designing a learning rate and penalty schedule for a wind farm
layout optimizer. Write ONLY `schedule_fn(step, total_steps, lr0, alpha0)`
that returns (lr, alpha, beta1, beta2).

## Rules
- Write schedule files to `{self.config.output_dir}/iter_NNN.py`
- Each file must define ONLY `schedule_fn(step, total_steps, lr0, alpha0)`
- Do NOT write `optimize()` — it will be rejected
- Score: `python tools/run_optimizer.py <script> --schedule-only`
- Test: `python tools/run_tests.py <script> --quick`
- Generalize: `python tools/test_generalization.py <script> --schedule-only`
- Status: `python tools/get_status.py --log {self.log_path}`
- Baseline: {self._get_baseline_aep():.1f} GWh
- Timeout: {self.config.timeout_per_run}s per run

## Workflow
1. Read `{self.config.output_dir}/agent_memory.md` for status
2. Read `results/seed_schedule.py` to see the starting schedule
3. Read `playground/skeleton.py` to understand the fixed optimizer
4. Write a new schedule, score it, iterate
"""

    def _write_memory_file(self):
        """Write agent_memory.md to the output dir."""
        from .memory import render_agent_memory
        content = render_agent_memory(
            self.session, self.history, self.attempts,
            output_dir=self.config.output_dir,
            mode=self.config.taxonomy_mode,
        )
        Path(self.memory_path).write_text(content)

    def _invoke_opencode(self, prompt: str) -> str:
        """Run a single `opencode run` invocation and return stdout."""
        cmd = [
            "opencode", "run",
            "--dangerously-skip-permissions",
            "-m", f"openai/{self._vllm_model_id}",
            prompt,
        ]

        env = {
            **os.environ,
            "PYTHONPATH": self.config.pythonpath,
            "JAX_ENABLE_X64": "True",
            # Redirect OpenAI provider to local vLLM server
            "OPENAI_API_KEY": "dummy",
            "OPENAI_BASE_URL": f"{self.base_url}/v1",
        }

        timeout = min(self.max_turns_per_iter * 120, int(self.time_remaining()) + 60)

        print(f"[opencode run] Invoking, {self.time_remaining()/60:.0f} min remaining...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=timeout, env=env,
                cwd=os.getcwd()
            )
            if result.returncode != 0:
                print(f"[opencode run] Non-zero exit: {result.returncode}")
                if result.stderr:
                    print(f"[opencode run] stderr: {result.stderr[:500]}")
            return result.stdout
        except subprocess.TimeoutExpired:
            print(f"[opencode run] Timed out after {timeout}s")
            return ""

    def _get_vllm_model_id(self) -> str:
        """Query the vLLM server for the actual served model ID."""
        import requests
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
            models = resp.json().get("data", [])
            if models:
                return models[0]["id"]
        except Exception:
            pass
        # Fallback to config model name
        return self.model.split("/", 1)[-1] if "/" in self.model else self.model

    def run(self):
        """Main loop: repeated `opencode run` invocations with memory updates."""
        self.start_time = time.time()
        self.session.start_time = self.start_time

        # Query actual model ID from vLLM and write OpenCode config
        vllm_model_id = self._get_vllm_model_id()
        print(f"[opencode] vLLM serving: {vllm_model_id}")
        self._write_opencode_config(vllm_model_id)

        # Determine iteration count
        if self.iterations > 0:
            n_iters = self.iterations
        else:
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

            self._write_memory_file()

            if i == 0:
                prompt = (
                    f"Read {self.memory_path} for context. Explore the pixwake codebase "
                    f"in playground/pixwake/src/, understand the API, then write and "
                    f"test an optimizer. Score it and test generalization.{seed_context}"
                )
            else:
                prompt = (
                    f"Read {self.memory_path} for your updated status and history. "
                    f"You have {self.time_remaining()/60:.0f} minutes remaining. "
                    f"Write an improved optimizer based on what you've learned. "
                    f"Try a different strategy than your last attempt. "
                    f"Test it, score it, and check generalization."
                )

            output = self._invoke_opencode(prompt)

            if output:
                print(f"[iter {i+1}] OpenCode output ({len(output)} chars):")
                lines = output.strip().split("\n")
                for line in lines[-10:]:
                    print(f"  {line}")

            # Sync attempt log
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
