"""Gemini mutation engine — the existing Gemini CLI subprocess path (spec 3.3ii).

Mirrors the v1 `gemini -p` subprocess integration. Logs the resolved model
string + token counts. NOT invoked for real in the dry run (the MockEngine
stands in); this module provides the wiring + provenance only.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time

from .base import Engine, EvoContext, MutationLog, MutationResult

DEFAULT_MODEL = "gemini-2.5-flash"
# rough public price (USD / 1M tokens)
_PRICE = {"gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-pro": (1.25, 10.0)}


class GeminiCLIEngine(Engine):
    name = "gemini_cli"

    def __init__(self, model: str = DEFAULT_MODEL, binary: str = "gemini",
                 timeout_s: int = 180, cwd: str | None = None):
        self.model = model
        self.binary = binary
        self.timeout_s = timeout_s
        # LAUNCH-GATE scoping: run the CLI with cwd in the scoped clean-room dir
        # (outside the repo tree), so any file the CLI reads resolves inside the
        # scope and cannot reach results/ paper/ specs/ prereg/ state/.
        self.cwd = cwd

    def preflight(self) -> None:
        if shutil.which(self.binary) is None:
            raise RuntimeError(
                f"Gemini CLI binary '{self.binary}' not found on PATH. Install/"
                f"authenticate the Gemini CLI before running the Gemini engine.")

    def _usd(self, pt: int, ct: int) -> float:
        pin, pout = _PRICE.get(self.model, (0.30, 2.50))
        return pt / 1e6 * pin + ct / 1e6 * pout

    def mutate(self, ctx: EvoContext) -> MutationResult:
        """Propose a child schedule via `gemini -p`. NOT called in the dry run."""
        self.preflight()
        t0 = time.time()
        prompt = _build_prompt(ctx)
        proc = subprocess.run(
            [self.binary, "-m", self.model, "-p", prompt],
            capture_output=True, text=True, timeout=self.timeout_s,
            cwd=self.cwd or None)
        out = proc.stdout or ""
        child = _extract_code(out)
        # token accounting: parse CLI usage line if present, else estimate ~4 chars/token
        pt, ct = _parse_tokens(out, prompt)
        usd = self._usd(pt, ct)
        log = MutationLog(engine=self.name, model=self.model,
                          prompt_tokens=pt, completion_tokens=ct, usd=round(usd, 6),
                          walltime_s=round(time.time() - t0, 3),
                          ok=child is not None,
                          error="" if child else (proc.stderr or "no code")[:200])
        return MutationResult(source=child, log=log)


def _build_prompt(ctx: EvoContext) -> str:
    fit = "\n".join(f"  {c}: {v}" for c, v in sorted(ctx.per_cell_fitness.items()))
    return (
        "Evolve this TopFarm-SGD schedule_fn(step, total_steps, D, min_spacing, "
        "n_turbines, gamma_min, alpha0). Return ONLY a Python code block defining "
        "schedule_fn.\n\n```python\n" + ctx.parent_source + "\n```\n\n"
        f"Parent per-cell fitness (%-over-baseline; feasibility only):\n{fit}\n")


def _parse_tokens(out: str, prompt: str):
    m = re.search(r"prompt[_ ]tokens?[:=]\s*(\d+).*?(?:completion|output)[_ ]tokens?[:=]\s*(\d+)",
                  out, re.I | re.S)
    if m:
        return int(m.group(1)), int(m.group(2))
    return max(1, len(prompt) // 4), max(1, len(out) // 4)   # ~4 chars/token estimate


def _extract_code(text: str):
    if not text:
        return None
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip() or None
    return text.strip() or None
