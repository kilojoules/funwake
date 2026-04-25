#!/usr/bin/env bash
# Experiment G: one additional frontier model under the narrow interface.
#
# Defaults to Claude Opus via the existing claude-code runner. Override
# MODEL or PROVIDER for other targets.
#
# Pre-launch:
#   - The chosen model is supported by the chosen runner.
#   - The hot-start file exists.
#   - Quota for ~5–8 hr of model usage.

set -euo pipefail
cd "$(dirname "$0")/../.."

PROVIDER=${PROVIDER:-claude-code}
MODEL=${MODEL:-claude-opus-4-7}
TIME_BUDGET=${TIME_BUDGET:-18000}
WIND_CSV=${WIND_CSV:-~/clusters/energy_island_10y_daily_av_wind.csv}
HOT_START=${HOT_START:-results/seed_schedule.py}
OUT=${OUT:-results_agent_${MODEL//[\/.:-]/_}_sched}

mkdir -p "$OUT"

[[ -f "$OUT/.done" ]] && { echo "[G] $OUT already done."; exit 0; }

pixi run python agent_cli.py \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --schedule-only \
    --wind-csv "$WIND_CSV" \
    --time-budget "$TIME_BUDGET" \
    --hot-start "$HOT_START" \
    --output-dir "$OUT" \
    --timeout-per-run 180 \
    > "$OUT/launch.log" 2>&1

touch "$OUT/.done"
echo "[G] Done. Output: $OUT"
