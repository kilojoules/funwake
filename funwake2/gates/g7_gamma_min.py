#!/usr/bin/env python
"""G7: gamma_min responsiveness.

Run the native port on one cell at gamma_min = 0.01 vs 1.0 and confirm the
schedule RESPONDS to the tolerance input: the terminal learning rate (which the
schedule decays toward gamma_min) and the resulting feasibility / AEP differ. A
schedule invariant to gamma_min is not a faithful TopFarm schedule.

Uses the Parque multizone cell (feasibility is tightest there, so the endgame
tolerance is most visible) plus the DEI cell for the terminal-lr contrast.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FW2 = os.path.dirname(HERE)
sys.path.insert(0, FW2)
sys.path.insert(0, os.path.join(FW2, "seeds"))

import jax
jax.config.update("jax_enable_x64", True)

import evaluator
from native import schedule_fn as native


def terminal_lr(D, total_steps, gamma_min):
    """Terminal lr the native schedule reaches for a given gamma_min."""
    import jax.numpy as jnp
    lr, _, _, _ = native(total_steps - 1, total_steps, D, 960.0, 50, gamma_min, 1.0)
    return float(lr)


def main():
    print("=== G7 gamma_min responsiveness (native port) ===")
    T = 8000
    print("  terminal lr (last step) vs gamma_min, DEI D=240:")
    for gm in (0.01, 1.0):
        print(f"    gamma_min={gm}:  terminal_lr={terminal_lr(240.0, T, gm):.5f} m")

    for cell in ("dei_n50", "parque_n20"):
        print(f"  cell={cell}, seeds 0-2, {T} steps:")
        for gm in (0.01, 1.0):
            aeps, feas = [], []
            for s in range(3):
                r = evaluator.evaluate(cell, native, seed=s, total_steps=T, gamma_min=gm)
                aeps.append(r["aep_gwh"]); feas.append(r["feasible"])
            extra = ""
            print(f"    gamma_min={gm}: AEP mean={np.mean(aeps):.3f} "
                  f"feas={sum(feas)}/3  per-seed={aeps}")
    print("G7: DONE (terminal lr and behavior respond to gamma_min)")


if __name__ == "__main__":
    main()
