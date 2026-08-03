"""Antigravity mutation engine — Google's Antigravity agent CLI (`agy`) as a
FunWake-2 mutation operator, used for the Gemini-family runs (the standalone
`gemini` CLI's free Code-Assist OAuth tier was discontinued; Antigravity's login
is the working path to Gemini 3.x here).

Same `Engine` interface as the Claude/Codex/Gemini engines: given a parent schedule
+ firewall-safe feedback, return an improved `schedule_fn` module. Drives `agy
--print` (single-prompt, non-interactive) with cwd pinned to the scoped clean-room
workspace, so the agent only ever sees the sanitized harness/seeds/feedback — never
results/ holdout/ pre-registration. No tools are enabled (the prompt forbids them).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time

from .base import Engine, EvoContext, MutationLog, MutationResult

DEFAULT_MODEL = "Gemini 3.1 Pro (High)"
# provenance estimate only (Antigravity subscription; not a billed amount)
_PRICE = {"Gemini 3.1 Pro (High)": (2.0, 12.0), "Gemini 3.6 Flash (High)": (0.3, 2.5)}


def _resolve_binary(binary: str) -> str | None:
    if os.path.sep in binary:
        return binary if os.path.exists(binary) else None
    found = shutil.which(binary)
    if found:
        return found
    fallback = os.path.expanduser("~/.local/bin/agy")
    return fallback if os.path.exists(fallback) else None


class AntigravityCLIEngine(Engine):
    name = "antigravity_cli"

    def __init__(self, model: str = DEFAULT_MODEL, binary: str = "agy",
                 timeout_s: int = 300, cwd: str | None = None):
        self.model = model
        self.binary = binary
        self.timeout_s = timeout_s
        self.cwd = cwd

    def preflight(self) -> None:
        if _resolve_binary(self.binary) is None:
            raise RuntimeError(
                f"Antigravity CLI '{self.binary}' not found (looked on PATH and "
                f"~/.local/bin/agy). Install/login to Antigravity first (`agy models` "
                f"should list models).")

    def _usd(self, pt: int, ct: int) -> float:
        pin, pout = _PRICE.get(self.model, (2.0, 12.0))
        return pt / 1e6 * pin + ct / 1e6 * pout

    def mutate(self, ctx: EvoContext) -> MutationResult:
        self.preflight()
        t0 = time.time()
        prompt = _build_prompt(ctx)
        binpath = _resolve_binary(self.binary)
        # --print: single non-interactive prompt; no tools enabled (prompt forbids
        # them). print-timeout slightly under the subprocess timeout as a soft cap.
        inner = max(30, self.timeout_s - 20)
        cmd = [binpath, "--print", prompt, "--model", self.model,
               "--print-timeout", f"{inner}s"]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=self.timeout_s, cwd=self.cwd or None)
        out = proc.stdout or ""
        child = _extract_code(out)
        pt, ct = max(1, len(prompt) // 4), max(1, len(out) // 4)   # ~4 chars/token
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


def _extract_code(text: str):
    if not text:
        return None
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip() or None
    if "def schedule_fn" in text:
        return text.strip()
    return None
