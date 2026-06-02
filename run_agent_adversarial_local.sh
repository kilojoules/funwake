#!/bin/bash
# Local adversarial-selection schedule-only agent run.
#
# Mirror of lumi/run_agent_adversarial.sbatch but for the user's local
# machine. Memory peak measured at ~470 MB harness + ~300 MB codex
# = ~820 MB concurrent (fits the user's 1 GB budget with ~180 MB margin).
#
# Usage:
#   bash run_agent_adversarial_local.sh
#   bash run_agent_adversarial_local.sh 7200   # 2-hour run instead of 5-hour
#
# Requires:
#   - codex CLI on PATH (verified: codex-cli 0.125.0)
#   - playground/pixwake cloned (this script will clone if missing)
#   - pixi env set up (pixi install)

set -e
cd "$(dirname "$0")"

TIME_BUDGET=${1:-18000}  # default 5 hours

# Ensure pixwake is cloned
if [ ! -d "playground/pixwake" ]; then
    echo "Cloning pixwake..."
    mkdir -p playground
    git clone https://github.com/kilojoules/cluster-tradeoffs.git playground/pixwake
    cd playground/pixwake && git checkout b8e905a && cd ../..
fi

# Memory hygiene: don't preallocate JAX, force CPU.
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

STRESS_PROBLEM="results/matrix/problem_dei_n50_roseuniform.json"
STRESS_BASELINE=5599.79
OUTDIR="results_agent_schedule_adversarial_local"
WIND_CSV="$HOME/clusters/energy_island_10y_daily_av_wind.csv"

echo "=== FunWake adversarial agent (local) ==="
echo "Start:        $(date)"
echo "Time budget:  ${TIME_BUDGET}s ($(($TIME_BUDGET/60)) min)"
echo "Stress:       $STRESS_PROBLEM (baseline $STRESS_BASELINE GWh)"
echo "Output:       $OUTDIR"
echo "Wind CSV:     $WIND_CSV"
echo ""

pixi run python agent_cli.py \
    --provider codex \
    --model "${MODEL:-gpt-5.5}" \
    --schedule-only \
    --wind-csv "$WIND_CSV" \
    --hot-start results/seed_schedule.py \
    --output-dir "$OUTDIR" \
    --time-budget "$TIME_BUDGET" \
    --timeout-per-run 180 \
    --adversarial-stress "$STRESS_PROBLEM" \
    --adversarial-stress-baseline "$STRESS_BASELINE"

echo ""
echo "Done:         $(date)"
echo "Saved scripts: $(ls $OUTDIR/iter_*.py 2>/dev/null | wc -l)"
