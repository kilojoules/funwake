"""FunWake-2 controller configuration (frozen bins + run knobs).

The MAP-elites bins are FROZEN by the sign-off addendum (D-8) and must match
the pre-registration verbatim:

    peak_lr/D     : {<0.5, 0.5-0.8, 0.8-1.2, >1.2}
    terminal_lr_m : {<=0.01, 0.01-0.1, 0.1-1, >1}
    coupling      : {coupled, decoupled, cyclic}
    restarts      : {0, 1-2, >=3}
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# ── FROZEN MAP-elites bin edges (D-8) ─────────────────────────────────
PEAK_LR_OVER_D_EDGES = [0.5, 0.8, 1.2]          # 4 bins: <0.5, .5-.8, .8-1.2, >1.2
TERMINAL_LR_M_EDGES = [0.01, 0.1, 1.0]          # 4 bins: <=.01, .01-.1, .1-1, >1
COUPLING_CLASSES = ["coupled", "decoupled", "cyclic"]
RESTART_EDGES = [1, 3]                           # 3 bins: 0, 1-2, >=3 (count >= edge)

FEATURE_DIMENSIONS = ["peak_lr_over_D", "terminal_lr_m", "coupling", "restarts"]

# ── canonical evaluation constants ────────────────────────────────────
GAMMA_MIN = 0.01            # constraint tolerance (m) used in-search
GAMMA_MIN_RESPONSIVENESS = 1.0   # stage-C responsiveness probe (D-7)
TOTAL_STEPS = 8000         # D-1

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class RunConfig:
    """One evolutionary run's configuration + budget ceiling."""
    run_id: str = "dry_run"
    state_dir: str = "funwake2/state/controller"          # archive/RNG/meta
    cache_dir: str = "funwake2/state/eval_cache"          # content-addressed results
    lineage_path: str = "funwake2/state/lineage.jsonl"    # append-only JSONL

    # evaluation
    gamma_min: float = GAMMA_MIN
    total_steps: int = TOTAL_STEPS
    stage_a_seeds: List[int] = field(default_factory=lambda: [0, 1])
    stage_b_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    stage_c_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    noise_floor_gwh: float = 3.0    # stage-A reject margin (GWh, DEI-scale)

    # ── stage-B+ (elite-tier, gbar ONLY — never in the Mac evolution loop) ──
    # Top-k archive elites per generation re-scored on expensive high-N cells
    # (incl. n200) with 2-3 paired seeds. Disabled on the Mac; run on gbar where
    # ~5-min evals are affordable.
    enable_stage_b_plus: bool = False          # gbar sets True
    stage_b_plus_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    stage_b_plus_top_k: int = 3
    stage_b_plus_cells: List[str] = field(default_factory=list)   # e.g. ["dei_n200_rosedei"]

    # ── high-N stage-B cell — FROZEN = DEI n120 (DEI rose) ────────────
    # Chosen by the timing curve (native@c*D, 8000 steps, 1 seed): n80=72s,
    # n100=116s, n120=159s (<=3 min, feasible), n150=228s (rejected). Replaces
    # n200 in stage-B (n200 -> stage-B+ elite tier, gbar-only).
    high_n_cell: Optional[str] = "dei_n120_rosedei"

    # ── ops: heartbeat + raised watchdog for long evals ───────────────
    heartbeat: bool = True
    watchdog_seconds: int = 900     # raised so n80/n100-class evals don't trip it

    # stage-C holdout margin floor (GWh): ROWP measured floor in the real run,
    # a cheap-cell stand-in (~0.1) in the dry run.
    holdout_floor_gwh: float = 0.1

    # search
    num_islands: int = 2
    generations: int = 30
    proposals_per_gen: int = 20
    novelty_threshold: float = 0.92  # reject if code-similarity above this

    # budget ceiling (spec 5.4). Abort cleanly at 90% of EITHER.
    max_usd: float = 50.0
    max_tokens: int = 7_000_000
    abort_fraction: float = 0.90

    # dry-run overrides (machinery validation, MOCK mutator)
    dry_run: bool = False
    dry_cells: Optional[List[str]] = None    # smoke cells for both A and B

    def as_dict(self) -> Dict:
        return asdict(self)
