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
        "You are evolving a TopFarm-SGD learning-rate/penalty schedule for a wind-"
        "farm layout optimizer. A fixed Adam skeleton calls your function each step "
        "and does the rest; you ONLY choose (lr, alpha, beta1, beta2).\n\n"
        "Return ONE improved, self-contained Python module that defines ONLY:\n"
        "  def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):\n"
        "      ... ; return lr, alpha, beta1, beta2\n"
        "Rules: build the exploration learning rate from the rotor diameter D (no "
        "hardcoded lr0); decay lr toward gamma_min (the metre-valued tolerance); "
        "alpha0 = mean|grad J|/D is supplied; use jax.numpy (import jax.numpy as jnp) "
        "and keep it traceable (no Python branches on step). IMPORTANT: do NOT call "
        "float()/int() on step or alpha0 — they are traced JAX values inside the jit "
        "loop; use jnp arithmetic (jnp.asarray to cast). Output ONLY the code in a "
        "single ```python block; do not run anything or use any tools.\n\n"
        f"Parent schedule (generation {ctx.generation}):\n```python\n"
        + ctx.parent_source + "\n```\n\n"
        f"Parent per-cell fitness (%-over-baseline; feasibility booleans; AEP "
        f"firewalled):\n{fit}\n\n{ctx.notes}\n")


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
