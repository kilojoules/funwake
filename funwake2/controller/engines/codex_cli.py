"""Codex mutation engine — OpenAI Codex CLI as a FunWake-2 mutation operator.

Implements the same `Engine` interface as the Claude/Gemini engines: given a
parent schedule + firewall-safe feedback, return an improved `schedule_fn` module.
Runs `codex exec` non-interactively in a READ-ONLY sandbox, with cwd pinned to the
scoped clean-room workspace so Codex can only see the sanitized harness/seeds/
feedback (never results/ holdout/ pre-registration). Logs the resolved model +
token/cost estimate for provenance, exactly like the other engines.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time

from .base import Engine, EvoContext, MutationLog, MutationResult

DEFAULT_MODEL = "gpt-5.5"
# rough public price (USD / 1M tokens); estimate only — Codex CLI uses the
# ChatGPT subscription, so this is a provenance figure, not a billed amount.
_PRICE = {"gpt-5.5": (1.25, 10.0)}


class CodexCLIEngine(Engine):
    name = "codex_cli"

    def __init__(self, model: str = DEFAULT_MODEL, binary: str = "codex",
                 timeout_s: int = 300, cwd: str | None = None):
        self.model = model
        self.binary = binary
        self.timeout_s = timeout_s
        self.cwd = cwd

    def preflight(self) -> None:
        if shutil.which(self.binary) is None:
            raise RuntimeError(
                f"Codex CLI binary '{self.binary}' not found on PATH. Install/"
                f"authenticate the Codex CLI before running the Codex engine.")
        if not os.path.exists(os.path.expanduser("~/.codex/auth.json")):
            raise RuntimeError(
                "~/.codex/auth.json not found — run `codex login` first.")

    def _usd(self, pt: int, ct: int) -> float:
        pin, pout = _PRICE.get(self.model, (1.25, 10.0))
        return pt / 1e6 * pin + ct / 1e6 * pout

    def mutate(self, ctx: EvoContext) -> MutationResult:
        self.preflight()
        t0 = time.time()
        prompt = _build_prompt(ctx)
        fd, outfile = tempfile.mkstemp(suffix=".txt", prefix="codex_out_")
        os.close(fd)
        cmd = [self.binary, "exec", "--skip-git-repo-check",
               "-s", "read-only", "-m", self.model, "-o", outfile]
        if self.cwd:
            cmd += ["-C", self.cwd]
        cmd += [prompt]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self.timeout_s)
            last = ""
            if os.path.exists(outfile):
                with open(outfile) as f:
                    last = f.read()
        finally:
            if os.path.exists(outfile):
                os.unlink(outfile)
        text = last or proc.stdout or ""
        child = _extract_code(text)
        pt, ct = max(1, len(prompt) // 4), max(1, len(text) // 4)   # ~4 chars/token
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
        "loop; use jnp arithmetic (jnp.asarray to cast). Output ONLY the code in "
        "a single ```python block; do not run anything.\n\n"
        f"Parent schedule (generation {ctx.generation}):\n```python\n"
        + ctx.parent_source + "\n```\n\n"
        f"Parent per-cell fitness (%-over-baseline; feasibility booleans; higher is "
        f"better; AEP firewalled):\n{fit}\n\n{ctx.notes}\n")


def _extract_code(text: str):
    if not text:
        return None
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip() or None
    # fall back: if the whole message looks like a module, take it
    if "def schedule_fn" in text:
        return text.strip()
    return None
