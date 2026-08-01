#!/usr/bin/env python
"""G3 (BLOCKING): incumbent-port bit-identity at D=240.

For a fixed alpha0 input and D=240, assert
  iterXXX_port(step, 8000, D=240, ms, N, gamma_min, alpha0)
   == archived iter_XXX(step, 8000, lr0=50, alpha0)
for ALL steps and ALL four outputs (lr, alpha, beta1, beta2): max abs diff must
be EXACTLY 0. Also reports iter192_port peak_lr/D at DEI (D=240) and ROWP (D=198).
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

from _port import make_port  # noqa: E402

PORTS = {
    "iter192": "runs/schedule_only_5hr/iter_192.py",
    "iter181": "runs/schedule_only_5hr/iter_181.py",
    "iter118": "runs/gemini_cli_5hr/iter_118.py",
}

TOTAL = 8000
D_DEI = 240.0
MS, N, GM = 960.0, 50, 0.01
ALPHA0_VALUES = [1.0, 0.37, 12.5, 1e-3]   # a few fixed alpha0 inputs


def outputs(fn, args_builder):
    lr = np.empty(TOTAL); al = np.empty(TOTAL)
    b1 = np.empty(TOTAL); b2 = np.empty(TOTAL)
    for i in range(TOTAL):
        o = fn(*args_builder(i))
        lr[i], al[i], b1[i], b2[i] = (float(o[0]), float(o[1]),
                                      float(o[2]), float(o[3]))
    return lr, al, b1, b2


def main():
    print("=== G3 incumbent-port bit-identity (D=240) ===")
    all_pass = True
    peak_report = {}
    for name, rel in PORTS.items():
        port, archived = make_port(rel)
        worst = 0.0
        for a0 in ALPHA0_VALUES:
            po = outputs(port, lambda i: (i, TOTAL, D_DEI, MS, N, GM, a0))
            ar = outputs(archived, lambda i: (i, TOTAL, 50.0, a0))
            d = max(float(np.max(np.abs(p - r))) for p, r in zip(po, ar))
            worst = max(worst, d)
        ok = (worst == 0.0)
        all_pass = all_pass and ok
        print(f"  {name}: max_abs_diff over 4 outputs x {len(ALPHA0_VALUES)} "
              f"alpha0 x {TOTAL} steps = {worst:.3e}  -> {'PASS' if ok else 'FAIL'}")
        peak_report[name] = worst

    # peak_lr/D for iter192 at DEI (D=240) and ROWP (D=198)
    port192, _ = make_port(PORTS["iter192"])
    for label, Dv in (("DEI D=240", 240.0), ("ROWP D=198", 198.0)):
        lrs = np.array([float(port192(i, TOTAL, Dv, MS, N, GM, 1.0)[0])
                        for i in range(TOTAL)])
        lr0_internal = (50.0 / 240.0) * Dv
        print(f"  iter192_port {label}: internal lr0={lr0_internal:.4f}  "
              f"peak_lr={lrs.max():.4f}  peak_lr/D={lrs.max()/Dv:.4f}")

    print(f"G3: {'PASS' if all_pass else 'FAIL'} "
          f"(all incumbent ports bit-identical at D=240)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
