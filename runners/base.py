"""Base runner — shared config, attempt logging, and scoring logic.

Both GeminiRunner and ClaudeCodeRunner inherit from this.
"""
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RunConfig:
    """Shared configuration for all backends."""
    wind_csv: str
    time_budget: int = 3600          # seconds
    output_dir: str = "results_agent"
    train_problem: str = "playground/problem.json"
    rowp_problem: str = "results/problem_rowp.json"
    baselines: str = "results/baselines.json"
    train_farm: str = "1"
    hot_start: Optional[str] = None  # seed optimizer path
    timeout_per_run: int = 60        # per-optimizer timeout
    pixwake_src: str = "playground/pixwake/src"

    # Phase-2 / diversity nudge thresholds
    phase2_fraction: float = 0.3     # switch to custom optimizer nudge
    diversity_after: int = 5         # nudge after N consecutive sgd_solve

    @property
    def pythonpath(self):
        return f"{self.pixwake_src}:{os.environ.get('PYTHONPATH', '')}"


@dataclass
class AttemptResult:
    attempt: int
    timestamp: float
    train_aep: Optional[float] = None
    train_feasible: Optional[bool] = None
    train_time: Optional[float] = None
    train_baseline: Optional[float] = None
    rowp_aep: Optional[float] = None       # only stored in log, never shown to LLM
    rowp_feasible: Optional[bool] = None
    rowp_time: Optional[float] = None
    strategy: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self):
        d = {"attempt": self.attempt, "timestamp": self.timestamp}
        for k in ["train_aep", "train_feasible", "train_time", "train_baseline",
                   "rowp_aep", "rowp_feasible", "rowp_time", "strategy", "error"]:
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


class BaseRunner(ABC):
    """Shared logic for the agent loop."""

    def __init__(self, config: RunConfig):
        from .memory import SessionState, HistoryLog, TranscriptStore

        self.config = config
        self.attempts: list[AttemptResult] = []
        self.best_aep: float = -float("inf")
        self.best_script: Optional[str] = None
        self.start_time: float = 0

        # Memory scaffolding
        self.history = HistoryLog()
        self.transcript = TranscriptStore()
        self.session = SessionState(
            session_id=f"{config.output_dir}_{int(time.time())}",
            start_time=0,  # set in run()
            time_budget=config.time_budget,
            baseline_aep=self._get_baseline_aep(),
        )

        # Setup output directory
        os.makedirs(config.output_dir, exist_ok=True)
        self.log_path = os.path.join(config.output_dir, "attempt_log.json")
        self.session_path = os.path.join(config.output_dir, "session.json")
        self.memory_path = "agent_memory.md"

        # Load existing log if resuming
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                existing = json.load(f)
            self.attempts = existing  # raw dicts, fine for logging
            successes = [e for e in existing if "train_aep" in e]
            if successes:
                best = max(successes, key=lambda e: e["train_aep"])
                self.best_aep = best["train_aep"]

    def time_remaining(self) -> float:
        return max(0, self.config.time_budget - (time.time() - self.start_time))

    def elapsed_minutes(self) -> float:
        return (time.time() - self.start_time) / 60

    def in_phase2(self) -> bool:
        elapsed_frac = (time.time() - self.start_time) / self.config.time_budget
        return elapsed_frac > self.config.phase2_fraction

    def log_attempt(self, result: AttemptResult):
        """Append to attempt log, update session state and history."""
        from .memory import save_session, render_agent_memory
        from pathlib import Path

        self.attempts.append(result.to_dict())

        # Update session state
        self.session.attempts_total += 1
        if result.error:
            self.session.attempts_error += 1
            self.history.add("error", f"Attempt {result.attempt} failed",
                           result.error[:100])
        else:
            self.session.attempts_success += 1
            if result.strategy:
                self.session.strategies_tried.append(result.strategy)
                if result.strategy == "sgd_solve":
                    self.session.consecutive_sgd_solve += 1
                else:
                    self.session.consecutive_sgd_solve = 0

        if result.train_aep and result.train_aep > self.best_aep:
            self.best_aep = result.train_aep
            self.session.best_aep = result.train_aep
            self.session.best_iter = result.attempt
            self.session.best_strategy = result.strategy or ""
            self.history.add("milestone",
                           f"New best: {result.train_aep:.1f} GWh",
                           f"Attempt {result.attempt}, strategy: {result.strategy}")
            # Copy best script
            best_path = os.path.join(self.config.output_dir, "best_optimizer.py")
            src = os.path.join(self.config.output_dir,
                               f"iter_{result.attempt:03d}.py")
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, best_path)

        # Update phase
        if self.in_phase2():
            self.session.phase = "exploit"

        # Persist everything
        with open(self.log_path, "w") as f:
            json.dump(self.attempts, f, indent=2)
        save_session(self.session, Path(self.session_path))

        # Refresh agent_memory.md for Claude Code
        memory_md = render_agent_memory(self.session, self.history, self.attempts)
        Path(self.memory_path).write_text(memory_md)

    def build_system_prompt(self) -> str:
        """Build the system prompt / CLAUDE.md content. Shared across backends."""
        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            train_problem=self.config.train_problem,
            time_budget_min=self.config.time_budget // 60,
            timeout_per_run=self.config.timeout_per_run,
            baseline_aep=self._get_baseline_aep(),
        )
        if self.in_phase2():
            prompt += PHASE2_NUDGE
        return prompt

    def build_memory_context(self) -> str:
        """Summarize attempt history for the LLM."""
        if not self.attempts:
            return "No attempts yet."

        lines = [f"## Attempt History ({len(self.attempts)} total)\n"]
        recent = self.attempts[-10:]  # last 10
        for a in recent:
            if "error" in a:
                lines.append(f"- Attempt {a['attempt']}: ERROR — {a['error'][:100]}")
            elif "train_aep" in a:
                delta = ""
                if "train_baseline" in a:
                    d = a["train_aep"] - a["train_baseline"]
                    delta = f" (Δ={d:+.1f})"
                feas = "✓" if a.get("rowp_feasible") else "✗"
                lines.append(
                    f"- Attempt {a['attempt']}: AEP={a['train_aep']:.1f}{delta}, "
                    f"ROWP={feas}, time={a.get('train_time', '?'):.1f}s"
                )

        lines.append(f"\n**Best so far: {self.best_aep:.1f} GWh**")
        return "\n".join(lines)

    def _get_baseline_aep(self) -> float:
        try:
            with open(self.config.baselines) as f:
                baselines = json.load(f)
            return baselines[self.config.train_farm]["aep_gwh"]
        except (FileNotFoundError, KeyError):
            return 0.0

    @abstractmethod
    def run(self):
        """Main agent loop. Implemented by each backend."""
        ...


# ─── Shared prompt content ───────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert wind farm layout optimizer. Your task is to write a Python
`optimize()` function that maximizes Annual Energy Production (AEP) for
turbine layouts inside polygon boundaries.

## Interface

Your function signature MUST be:
```python
def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    # Returns (opt_x, opt_y) — JAX arrays of turbine positions
```

## Available tools

| Command | What it does |
|---------|-------------|
| `python tools/run_tests.py <script> {train_problem}` | Unit tests (signature, 3-turbine, full farm) |
| `python tools/run_optimizer.py <script>` | Score on training farm → AEP in GWh |
| `python tools/test_generalization.py <script>` | Held-out farm → PASS/FAIL only (no AEP) |
| `python tools/get_status.py` | Current best AEP vs baseline |

## Constraints

- Time budget: {time_budget_min} minutes total
- Each optimizer run times out at {timeout_per_run}s
- Baseline (500 multi-start SGD): {baseline_aep:.1f} GWh — beat this
- You can read files in `playground/` (pixwake source, problem.json)
- Write optimizers to the workspace directory
- You CANNOT modify the wake model, harness, or scorer

## Strategy tips

- Start by reading the problem JSON and pixwake source to understand the API
- The `topfarm_sgd_solve` function handles boundary/spacing constraints
- Multi-start with diverse initializations helps find better optima
- Consider how wind direction affects wake losses
- Test on the held-out farm periodically to check generalization
"""

PHASE2_NUDGE = """

## Phase 2: Custom optimizers

You've been running for a while. Consider going beyond `topfarm_sgd_solve`:
- Write a custom optimizer using `jax.grad` directly
- Try Adam, L-BFGS, or other optimizers
- Implement wind-direction-aware initialization
- Use the gradient information more creatively

Here's an Adam template to start from:
```python
import jax
import jax.numpy as jnp

def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    def objective(x, y):
        r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
        p = r.power()[:, :len(x)]
        return -jnp.sum(p * weights[:, None]) * 8760 / 1e6

    # Initialize layout, then:
    grad_fn = jax.grad(lambda params: objective(params[:n], params[n:]))
    # ... implement Adam update loop with boundary projection
```
"""
