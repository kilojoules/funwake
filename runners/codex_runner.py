"""Codex CLI backend — delegates to `codex exec` with sandbox bypass.

Architecture mirrors GeminiCLIRunner / ClaudeCodeRunner:
  Outer loop (this file): manages time budget, logging, memory context
  Inner loop (Codex CLI): reads code, writes optimizers, runs tools

Each iteration = one `codex exec` invocation. Between iterations, this
runner updates the memory file and re-invokes with fresh context.

Codex picks up its task instructions from AGENTS.md in the project
root (the upstream convention) plus the per-invocation prompt.
"""
import json
import os
import subprocess
import time
from pathlib import Path

from .base import BaseRunner, RunConfig, AttemptResult


CODEX_INSTRUCTION_FILE = "AGENTS.md"


class CodexRunner(BaseRunner):
    """Codex CLI agent via `codex exec` invocations."""

    def __init__(self, config: RunConfig,
                 model: str = "gpt-5.5",
                 iterations: int = 0,            # 0 = auto (fill time budget)
                 schedule_only: bool = False,
                 sandbox_mode: str = "workspace-write",
                 model_provider: str = None,
                 provider_base_url: str = None,
                 provider_env_key: str = "OPENAI_API_KEY",
                 provider_wire_api: str = "responses"):
        super().__init__(config)
        self.model = model
        self.iterations = iterations
        self.schedule_only = schedule_only
        self.sandbox_mode = sandbox_mode
        # Optional custom provider — injected via `codex -c model_providers.<name>=...`
        # so we don't have to mutate ~/.codex/config.toml.
        self.model_provider = model_provider
        self.provider_base_url = provider_base_url
        self.provider_env_key = provider_env_key
        self.provider_wire_api = provider_wire_api

        try:
            result = subprocess.run(["codex", "--version"],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise EnvironmentError("codex CLI returned non-zero on --version.")
        except FileNotFoundError:
            raise EnvironmentError("codex CLI not found in PATH. Install: see https://github.com/openai/codex")

    # -- task spec (AGENTS.md) -------------------------------------------------

    def _build_agents_md(self) -> str:
        if self.schedule_only:
            return self._build_schedule_agents_md()
        return f"""\
# FunWake Optimizer Agent

You are optimizing wind farm layouts. Write `optimize()` functions that
maximize AEP (Annual Energy Production).

## Rules
- Write optimizers to `{self.config.output_dir}/iter_NNN.py`.
- You MUST score every optimizer by running:
  `python tools/run_optimizer.py <script>`
- Run tests first: `python tools/run_tests.py <script> --quick`
- Test generalization: `python tools/test_generalization.py <script>`
- Check status: `python tools/get_status.py --log {self.log_path}`
- Do NOT modify files in `playground/pixwake/`, `benchmarks/`, or `tools/`.
- Do NOT run Python directly outside the tool scripts above.
- Baseline to beat: {self._get_baseline_aep():.1f} GWh.

## Timeout
Each optimizer run times out at {self.config.timeout_per_run}s.
Use 1-3 multi-starts, not 10+. Seed runs in ~23s.

## Function signature (must match exactly)
```python
def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    # Returns (opt_x, opt_y) — JAX arrays
```

## Key pixwake API
- `SGDSettings(learning_rate, max_iter, additional_constant_lr_iterations, tol, ...)`
- `topfarm_sgd_solve(objective, init_x, init_y, boundary, min_spacing, settings)`
- `sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None).power()` — shape (n_findex, n_turbines)

## Workflow
1. Read `{self.config.output_dir}/agent_memory.md` for status and history.
2. Read pixwake source if needed (`playground/pixwake/src/pixwake/`).
3. Write an optimizer script.
4. Run tests, then score, then test generalization.
5. Iterate: try different strategies (init, hyperparams, custom solvers).

## Environment
PYTHONPATH includes `playground/pixwake/src`. JAX_ENABLE_X64=True is set.
"""

    def _build_schedule_agents_md(self) -> str:
        adv_path = self.config.adversarial_stress
        adv_baseline = self.config.adversarial_stress_baseline
        if adv_path:
            score_cmd = (
                f"python tools/run_optimizer.py <script> --schedule-only "
                f"--stress-problem {adv_path}"
                + (f" --stress-baseline {adv_baseline}"
                   if adv_baseline is not None else "")
            )
            adv_section = f"""
## Adversarial selection (ACTIVE)

Every score returns BOTH `gap` (train) and `stress_gap`. The
deployment criterion is `min_gap = min(gap, stress_gap)` — deploy
only if `min_gap` improves over current best (see `best_min_gap`
in `get_status.py`).

Stress problem: {adv_path} (baseline {adv_baseline} GWh, same N=50
DEI polygon, different wind rose chosen because previous schedules
fail there). A schedule that wins on train but breaks on stress
will NOT be deployed.
"""
        else:
            score_cmd = "python tools/run_optimizer.py <script> --schedule-only"
            adv_section = ""

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
    # Returns (lr, alpha, beta1, beta2)
    ...
```

## Skeleton (fixed, do not change)
- Wind-direction-aware grid initialization inside the polygon.
- Computes gradients of AEP objective and constraint penalties (boundary + spacing).
- Combines: `grad = grad_obj + alpha * grad_constraint`.
- Adam update with your beta1, beta2 at your learning rate.
- Runs for 8000 steps total.

## Rules
- Write schedule files to `{self.config.output_dir}/iter_NNN.py`.
- NEVER overwrite an existing iter_NNN.py. Always increment N.
- Each file defines ONLY `schedule_fn(step, total_steps, lr0, alpha0)`.
- Do NOT write `optimize()`. Do NOT import topfarm_sgd_solve.
- Score: `{score_cmd}`
- Test: `python tools/run_tests.py <script> --quick`
- Generalize: `python tools/test_generalization.py <script> --schedule-only`
- Status: `python tools/get_status.py --log {self.log_path}`
- Baseline: {self._get_baseline_aep():.1f} GWh.
{adv_section}
## Workflow
1. Read `{self.config.output_dir}/agent_memory.md` for status.
2. Read `results/seed_schedule.py` for the starting schedule.
3. Read `playground/skeleton.py` to understand the fixed optimizer.
4. Write a new schedule, score it, iterate.
"""

    def _write_memory_file(self):
        from .memory import render_agent_memory
        content = render_agent_memory(
            self.session, self.history, self.attempts,
            output_dir=self.config.output_dir,
            mode=self.config.taxonomy_mode,
        )
        Path(self.memory_path).write_text(content)

    def _setup_codex_config(self):
        """Write AGENTS.md in the project root."""
        Path(CODEX_INSTRUCTION_FILE).write_text(self._build_agents_md())

    # -- inner loop ------------------------------------------------------------

    def _invoke_codex(self, prompt: str) -> str:
        cmd = ["codex"]
        if self.model_provider and self.provider_base_url:
            # Inline-define a custom provider for this exec call. Avoids
            # mutating ~/.codex/config.toml. Codex requires wire_api in
            # {"responses"} as of 0.125; for vLLM backends use a translator
            # (e.g. LiteLLM with use_chat_completions_api: true) in front.
            provider_block = (
                f'{self.model_provider}={{'
                f'name="{self.model_provider}",'
                f'base_url="{self.provider_base_url}",'
                f'env_key="{self.provider_env_key}",'
                f'wire_api="{self.provider_wire_api}"'
                f'}}'
            )
            cmd += [
                "-c", f"model_providers.{provider_block}",
                "-c", f"model_provider={self.model_provider}",
            ]
        cmd += [
            "exec",
            "--sandbox", self.sandbox_mode,
            "--skip-git-repo-check",
            "--color", "never",
            "-m", self.model,
            prompt,
        ]
        env = {
            **os.environ,
            "PYTHONPATH": self.config.pythonpath,
            "JAX_ENABLE_X64": "True",
        }
        timeout = int(self.time_remaining()) + 60

        print(f"[codex exec] model={self.model}, "
              f"{self.time_remaining()/60:.0f} min remaining...")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, env=env, cwd=os.getcwd(),
            )
            if result.returncode != 0:
                print(f"[codex exec] Non-zero exit: {result.returncode}")
                if result.stderr:
                    print(f"[codex exec] stderr: {result.stderr[:500]}")
            return result.stdout
        except subprocess.TimeoutExpired:
            print(f"[codex exec] Timed out after {timeout}s")
            return ""

    # -- main loop -------------------------------------------------------------

    def run(self):
        self.start_time = time.time()
        self.session.start_time = self.start_time
        self._setup_codex_config()

        if self.iterations > 0:
            n_iters = self.iterations
        else:
            # Codex sessions tend to be longer-form than gemini -p; budget ~10 min/iter.
            n_iters = max(1, self.config.time_budget // 600)

        seed_context = ""
        if self.config.hot_start and os.path.exists(self.config.hot_start):
            seed = Path(self.config.hot_start).read_text()
            seed_context = f"\n\nHere's a seed optimizer to start from:\n```python\n{seed}\n```"

        # Back-off so back-to-back fast failures don't hammer codex / trip
        # rate limits. Grows on consecutive sub-30s exits, resets after a
        # real-work iter.
        backoff_s = 0
        for i in range(n_iters):
            if self.time_remaining() <= 30:
                print(f"[iter {i+1}] Time's up.")
                break

            if backoff_s:
                print(f"[iter {i+1}] backing off {backoff_s}s after fast exit...")
                time.sleep(backoff_s)

            self._write_memory_file()

            mem_path = self.memory_path
            if i == 0:
                prompt = (
                    f"Read {CODEX_INSTRUCTION_FILE} and "
                    f"{mem_path} for context. Explore the pixwake codebase "
                    f"in playground/pixwake/src/, understand the API, then write "
                    f"and test an optimizer. Score it and test generalization."
                    f"{seed_context}"
                )
            else:
                prompt = (
                    f"Read {mem_path} for your updated status and history. "
                    f"You have {self.time_remaining()/60:.0f} minutes remaining. "
                    f"Write an improved optimizer based on what you've learned. "
                    f"Try a different strategy than your last attempt. "
                    f"Test it, score it, and check generalization."
                )

            iter_t0 = time.time()
            output = self._invoke_codex(prompt)
            iter_secs = time.time() - iter_t0

            if output:
                print(f"[iter {i+1}] codex output ({len(output)} chars). Tail:")
                for line in output.strip().split("\n")[-10:]:
                    print(f"  {line}")

            self._sync_attempts()
            print(f"[iter {i+1}] Attempts: {len(self.attempts)}, "
                  f"Best: {self.best_aep:.1f} GWh, "
                  f"Elapsed: {self.elapsed_minutes():.1f} min\n")

            # Fast-exit back-off: codex exec normally takes minutes per
            # call when it does real work. If it returned in <30s the
            # call almost certainly failed (rate limit / auth blip).
            # Double the wait each consecutive failure, capped at 5 min.
            if iter_secs < 30:
                backoff_s = min(max(backoff_s * 2, 30), 300)
            else:
                backoff_s = 0

        print(f"\nDone. Best AEP: {self.best_aep:.1f} GWh "
              f"over {len(self.attempts)} attempts.")

    def _sync_attempts(self):
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                self.attempts = json.load(f)
            successes = [a for a in self.attempts if "train_aep" in a]
            if successes:
                best = max(successes, key=lambda a: a["train_aep"])
                self.best_aep = best["train_aep"]
