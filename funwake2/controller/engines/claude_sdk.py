"""Claude mutation engine — Claude Agent SDK over the Max-plan OAuth credit.

Spec 3.3(i). HARD REQUIREMENTS (unit-tested, and NOT invoked for real in the dry
run):

  * **Preflight RAISES if ``ANTHROPIC_API_KEY`` is present** in the engine's
    environment, so a raw API key cannot shadow the subscription OAuth token and
    silently bill the metered API. Presence (not truthiness) is checked: even an
    empty ``ANTHROPIC_API_KEY=""`` shadows the OAuth profile, so it is rejected.
  * Auth is the **Claude Agent SDK path ONLY** — `claude setup-token` →
    ``CLAUDE_CODE_OAUTH_TOKEN`` consumed by ``claude_agent_sdk.query(...)``. We
    **never** construct a raw-API ``anthropic.Anthropic`` client on the OAuth
    token. (The Claude Agent SDK is a distinct product from the Messages-API SDK;
    docs: code.claude.com/docs/en/agent-sdk.)
  * Logs the resolved model string + prompt/completion tokens + $ per call.

This module does NOT import jax, anthropic, or claude_agent_sdk at import time
(all lazy), so the preflight unit test runs with nothing installed.
"""
from __future__ import annotations

import os
import time

from .base import Engine, EvoContext, MutationLog, MutationResult

# Opus-tier default per the claude-api skill mandate; overridable at construction.
DEFAULT_MODEL = "claude-opus-4-8"

# rough public price (USD / 1M tokens) for a $ estimate in the lineage log
_PRICE = {"claude-opus-4-8": (5.0, 25.0)}   # (input, output) per 1M


class AnthropicApiKeyPresentError(RuntimeError):
    """Raised by preflight when ANTHROPIC_API_KEY is set — it would shadow the
    subscription OAuth token and silently bill the metered API."""


class ClaudeSDKEngine(Engine):
    name = "claude_agent_sdk"

    def __init__(self, model: str = DEFAULT_MODEL, system_prompt: str = "",
                 cwd: str | None = None):
        self.model = model
        self.system_prompt = system_prompt
        # LAUNCH-GATE scoping: the SDK runs with cwd pointed at the scoped clean-
        # room workspace (harness + seeds + firewalled feedback only). Combined
        # with allowed_tools=[] the mutator cannot reach the source tree at all.
        self.cwd = cwd

    # ── HARD-REQUIREMENT preflight (unit-tested) ─────────────────────
    def preflight(self) -> None:
        if "ANTHROPIC_API_KEY" in os.environ:
            raise AnthropicApiKeyPresentError(
                "ANTHROPIC_API_KEY is set; it would shadow the subscription "
                "OAuth token (CLAUDE_CODE_OAUTH_TOKEN) and silently bill the "
                "metered API. Unset ANTHROPIC_API_KEY before running the Claude "
                "engine. (Even an empty value shadows the OAuth profile.)")
        if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            raise RuntimeError(
                "CLAUDE_CODE_OAUTH_TOKEN is not set. Run `claude setup-token` to "
                "authenticate the Claude Agent SDK against the Max-plan credit.")

    def _usd(self, pt: int, ct: int) -> float:
        pin, pout = _PRICE.get(self.model, (5.0, 25.0))
        return pt / 1e6 * pin + ct / 1e6 * pout

    def mutate(self, ctx: EvoContext) -> MutationResult:
        """Propose a child schedule via the Claude Agent SDK. NOT called in the
        dry run — the machinery is validated with the MockEngine instead."""
        self.preflight()
        t0 = time.time()
        # Lazy import so the preflight test needs nothing installed and so a raw
        # anthropic client is never even importable on this path.
        from claude_agent_sdk import query, ClaudeAgentOptions  # noqa: F401

        prompt = _build_prompt(ctx, self.system_prompt)
        opt_kwargs = dict(model=self.model, system_prompt=self.system_prompt,
                          allowed_tools=[], permission_mode="bypassPermissions")
        if self.cwd:                       # confine any file resolution to the scope
            opt_kwargs["cwd"] = self.cwd
        options = ClaudeAgentOptions(**opt_kwargs)
        child_src, pt, ct, resolved_model = _run_query(query, prompt, options, self.model)
        usd = self._usd(pt, ct)
        log = MutationLog(engine=self.name, model=resolved_model,
                          prompt_tokens=pt, completion_tokens=ct, usd=round(usd, 6),
                          walltime_s=round(time.time() - t0, 3), ok=child_src is not None,
                          error="" if child_src else "no schedule extracted")
        return MutationResult(source=child_src, log=log)


def _build_prompt(ctx: EvoContext, system_prompt: str) -> str:
    fit = "\n".join(f"  {c}: {v}" for c, v in sorted(ctx.per_cell_fitness.items()))
    return (
        "You are evolving a TopFarm-SGD `schedule_fn(step, total_steps, D, "
        "min_spacing, n_turbines, gamma_min, alpha0)`. Return ONE improved "
        "Python module defining only `schedule_fn`.\n\n"
        f"Parent (gen {ctx.generation}, island {ctx.island}):\n"
        "```python\n" + ctx.parent_source + "\n```\n\n"
        f"Parent per-cell fitness (%-over-baseline; feasibility booleans only, "
        f"AEP firewalled):\n{fit}\n\n"
        f"{ctx.notes}\n"
    )


def _run_query(query, prompt, options, model):
    """Drive the Agent SDK query() async generator, accumulate assistant text +
    token usage. Isolated for unit-test monkeypatching; never run in the dry
    run."""
    import asyncio

    async def _go():
        text_parts, pt, ct, resolved = [], 0, 0, model
        async for msg in query(prompt=prompt, options=options):
            # ResultMessage / AssistantMessage shapes carry usage + content.
            usage = getattr(msg, "usage", None)
            if usage:
                pt = int(getattr(usage, "input_tokens", pt) or pt)
                ct = int(getattr(usage, "output_tokens", ct) or ct)
            resolved = getattr(msg, "model", resolved) or resolved
            for block in getattr(msg, "content", []) or []:
                t = getattr(block, "text", None)
                if t:
                    text_parts.append(t)
        return "".join(text_parts), pt, ct, resolved

    text, pt, ct, resolved = asyncio.run(_go())
    return _extract_code(text), pt, ct, resolved


def _extract_code(text: str):
    if not text:
        return None
    if "```" in text:
        seg = text.split("```", 2)
        body = seg[1] if len(seg) > 1 else ""
        if body.startswith("python"):
            body = body[len("python"):]
        return body.strip() or None
    return text.strip() or None
