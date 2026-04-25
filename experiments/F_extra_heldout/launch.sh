#!/usr/bin/env bash
# Experiment F: synthetic L-shape held-out cell.
#
# Steps:
#   1. Build polygon JSON.
#   2. Run 500-multi-start baseline.
#   3. Score top Claude/Gemini schedules + bump-family DE on the new cell.

set -euo pipefail
cd "$(dirname "$0")/../.."

PROBLEM=results/problem_lshape.json
BASELINE=results/baseline_lshape.json

# Step 1: build polygon
if [[ ! -f "$PROBLEM" ]]; then
    pixi run python experiments/F_extra_heldout/build_polygon.py --output "$PROBLEM"
fi

# Step 2: 500-multi-start baseline (uses tools/compute_baselines_parallel.py)
if [[ ! -f "$BASELINE" ]]; then
    echo "[F] Running 500-multi-start baseline on L-shape (~6 hr CPU)."
    pixi run python tools/compute_baselines_parallel.py \
        --problems "$PROBLEM" \
        --n-starts 500 \
        --workers 10 \
        --output "$BASELINE"
fi

# Step 3: score top schedules
SCHEDULES=(
    "results_agent_schedule_only_5hr/iter_192.py"
    "results_agent_gemini_cli_5hr/iter_118.py"
    "results_bump_opt/best_bump.py"          # if exists
)

OUT=experiments/F_extra_heldout/results.json
mkdir -p experiments/F_extra_heldout
[[ -f "$OUT" ]] || echo '{}' > "$OUT"

for sched in "${SCHEDULES[@]}"; do
    [[ -f "$sched" ]] || { echo "[F] Skipping missing $sched"; continue; }
    echo "[F] Scoring $sched on L-shape..."
    pixi run python tools/multi_seed_eval.py "$sched" \
        --problem "$PROBLEM" \
        --n-seeds 3 \
        --output "experiments/F_extra_heldout/$(basename $sched .py)_lshape.json"
done

echo "[F] Done."
