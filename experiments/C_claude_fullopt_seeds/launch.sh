#!/usr/bin/env bash
# Experiment C: 4 additional Claude full-optimizer agent runs.
#
# Pre-launch checklist:
#   - ANTHROPIC_API_KEY set? Confirm a budget for ~5hr × 4 of Claude usage.
#   - results/seed_optimizer.py exists.
#   - Existing results_agent_claude_fullopt/ is preserved as run1.
#
# Usage:
#   bash experiments/C_claude_fullopt_seeds/launch.sh                 # runs 2..5 serial
#   PARALLEL=2 bash experiments/C_claude_fullopt_seeds/launch.sh      # 2 concurrent
#   RUNS="3 4" bash experiments/C_claude_fullopt_seeds/launch.sh      # subset

set -euo pipefail
cd "$(dirname "$0")/../.."

RUNS=${RUNS:-"2 3 4 5"}
PARALLEL=${PARALLEL:-1}
TIME_BUDGET=${TIME_BUDGET:-18000}     # 5 hr
WIND_CSV=${WIND_CSV:-~/clusters/energy_island_10y_daily_av_wind.csv}
HOT_START=${HOT_START:-results/seed_optimizer.py}

run_one() {
    local n="$1"
    local out=results_agent_claude_fullopt_run${n}
    if [[ -f "$out/.done" ]]; then
        echo "[C] $out already complete. Skipping."
        return
    fi
    mkdir -p "$out"
    pixi run python agent_cli.py \
        --provider claude-code \
        --wind-csv "$WIND_CSV" \
        --time-budget "$TIME_BUDGET" \
        --hot-start "$HOT_START" \
        --output-dir "$out" \
        --timeout-per-run 180 \
        > "$out/launch.log" 2>&1
    touch "$out/.done"
    echo "[C] run $n done."
}

export -f run_one

if [[ "$PARALLEL" -le 1 ]]; then
    for n in $RUNS; do run_one "$n"; done
else
    echo "$RUNS" | tr ' ' '\n' | xargs -P "$PARALLEL" -I{} bash -c 'run_one "$@"' _ {}
fi

echo "[C] Done."
