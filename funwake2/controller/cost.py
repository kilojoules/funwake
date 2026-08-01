"""Hard cost ceiling + clean 90% abort (spec 5.4).

The controller tracks cumulative $ and tokens from the per-invocation engine
logs and ABORTS CLEANLY at 90% of EITHER ceiling: it stops issuing new
mutations, finishes any in-flight eval, checkpoints, and stops. No run may
silently exceed budget.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class CostState:
    usd: float = 0.0
    tokens: int = 0
    n_calls: int = 0


class CostTracker:
    def __init__(self, max_usd: float, max_tokens: int,
                 abort_fraction: float = 0.90, state: CostState = None):
        self.max_usd = float(max_usd)
        self.max_tokens = int(max_tokens)
        self.abort_fraction = float(abort_fraction)
        self.state = state or CostState()

    def add(self, usd: float, tokens: int) -> None:
        self.state.usd += float(usd)
        self.state.tokens += int(tokens)
        self.state.n_calls += 1

    @property
    def usd_frac(self) -> float:
        return self.state.usd / self.max_usd if self.max_usd else 0.0

    @property
    def tokens_frac(self) -> float:
        return self.state.tokens / self.max_tokens if self.max_tokens else 0.0

    def should_abort(self) -> bool:
        """True once cumulative $ OR tokens reach 90% of the ceiling."""
        return (self.usd_frac >= self.abort_fraction
                or self.tokens_frac >= self.abort_fraction)

    def reason(self) -> str:
        if self.usd_frac >= self.abort_fraction:
            return (f"USD {self.state.usd:.4f}/{self.max_usd:.4f} "
                    f"= {self.usd_frac:.1%} >= {self.abort_fraction:.0%}")
        if self.tokens_frac >= self.abort_fraction:
            return (f"tokens {self.state.tokens}/{self.max_tokens} "
                    f"= {self.tokens_frac:.1%} >= {self.abort_fraction:.0%}")
        return ""

    def to_dict(self) -> dict:
        return {"state": asdict(self.state), "max_usd": self.max_usd,
                "max_tokens": self.max_tokens,
                "abort_fraction": self.abort_fraction}

    @classmethod
    def from_dict(cls, d: dict) -> "CostTracker":
        return cls(d["max_usd"], d["max_tokens"], d.get("abort_fraction", 0.9),
                   CostState(**d["state"]))
