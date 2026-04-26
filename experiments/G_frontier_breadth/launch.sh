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

# Default target: codex (since user has it running). Override for Opus etc.
PROVIDER=${PROVIDER:-codex}
MODEL=${MODEL:-gpt-5-codex}
MODE=${MODE:-sched}                       # sched | full
TIME_BUDGET=${TIME_BUDGET:-10800}          # 3 hr default for sched
WIND_CSV=${WIND_CSV:-~/clusters/energy_island_10y_daily_av_wind.csv}
if [[ "$MODE" == "sched" ]]; then
    HOT_START=${HOT_START:-results/seed_schedule.py}
    SCHED_FLAG=--schedule-only
    [[ "$TIME_BUDGET" -lt 10800 ]] && TIME_BUDGET=10800
else
    HOT_START=${HOT_START:-results/seed_optimizer.py}
    SCHED_FLAG=
    [[ "$TIME_BUDGET" -lt 16200 ]] && TIME_BUDGET=16200
fi

OUT=${OUT:-results_agent_${MODEL//[\/.:-]/_}_${MODE}}

mkdir -p "$OUT"
[[ -f "$OUT/.done" ]] && { echo "[G] $OUT already done."; exit 0; }

pixi run python agent_cli.py \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    $SCHED_FLAG \
    --wind-csv "$WIND_CSV" \
    --time-budget "$TIME_BUDGET" \
    --hot-start "$HOT_START" \
    --output-dir "$OUT" \
    --timeout-per-run 180 \
    > "$OUT/launch.log" 2>&1

touch "$OUT/.done"
echo "[G] Done. Output: $OUT"
