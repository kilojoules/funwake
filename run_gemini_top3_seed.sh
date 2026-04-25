#!/bin/bash
# Gemini CLI full-opt run seeded with the 3 best approaches.
#
# Usage:
#   bash run_gemini_top3_seed.sh
#
# Prerequisites:
#   - gemini CLI installed and authenticated (OAuth)
#   - pixi install done

set -e
cd "$(dirname "$0")"

OUTDIR="results_agent_gemini_top3_seed"
mkdir -p "$OUTDIR"

# Copy seeds into the output dir so the agent can see them
cp seeds_top3/seed_slsqp.py "$OUTDIR/"
cp seeds_top3/seed_bo_adam.py "$OUTDIR/"
cp seeds_top3/seed_dual_bumps_schedule.py "$OUTDIR/"

PROMPT="You are optimizing wind farm layouts. You have access to 3 proven approaches in $OUTDIR/:

1. seed_slsqp.py — SLSQP with JAX Jacobians (ROWP=4272.7 GWh, community standard)
2. seed_bo_adam.py — Successive Halving Adam with BO-tuned hex init (ROWP=4271.9 GWh, novel)
3. seed_dual_bumps_schedule.py — Schedule-only: dual Gaussian LR bumps (ROWP=4271.5 GWh, novel schedule)

These all reach a shared local optimum at ~4272 GWh on the held-out ROWP farm. Your goal: try to EXCEED this optimum.

Tools:
  pixi run python tools/run_optimizer.py <script> --timeout 180
  pixi run python tools/run_tests.py <script> --quick
  pixi run python tools/test_generalization.py <script>

Rules:
- Write optimize(sim, n_target, boundary, min_spacing, wd, ws, weights) functions
- NO import os, NO file I/O — all inputs via function args
- 180s timeout per evaluation
- Write scripts to $OUTDIR/iter_NNN.py

Start by reading all 3 seeds to understand their approaches, then try to combine or improve them. Ideas:
- Hybrid: SLSQP init + Adam refinement with bumpy schedule
- Novel constraint handling (augmented Lagrangian, interior point)
- Better initialization (Voronoi, Poisson disk, wind-aware clustering)
- Adaptive step sizes that respond to constraint violations"

echo "Starting Gemini CLI with top-3 seeds..."
echo "Output dir: $OUTDIR"
echo ""

gemini -p "$PROMPT"
