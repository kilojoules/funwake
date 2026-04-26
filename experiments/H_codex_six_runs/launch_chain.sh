#!/usr/bin/env bash
# Experiment H: 6 sequential codex runs (3 sched + 3 full).
#
# Sequential because Codex CLI single-account concurrency is 1.
# Total wall-clock estimate: 3 × 3hr + 3 × 4.5hr = ~22.5 hr.
#
# Resume-safe: each run writes .done on success. Re-running skips
# completed cells.
#
# Usage:
#   bash experiments/H_codex_six_runs/launch_chain.sh                 # all 6
#   MODES=sched bash experiments/H_codex_six_runs/launch_chain.sh     # only sched
#   SEEDS="1" bash experiments/H_codex_six_runs/launch_chain.sh       # only seed 1

set -euo pipefail
cd "$(dirname "$0")/../.."

MODES=${MODES:-"sched full"}
SEEDS=${SEEDS:-"1 2 3"}
WIND_CSV=${WIND_CSV:-~/clusters/energy_island_10y_daily_av_wind.csv}
CODEX_MODEL=${CODEX_MODEL:-gpt-5.5}
SCHED_BUDGET=${SCHED_BUDGET:-10800}    # 3 hr
FULL_BUDGET=${FULL_BUDGET:-16200}      # 4.5 hr

run_one() {
    local mode="$1"
    local seed="$2"

    local out hot_start budget sched_flag
    if [[ "$mode" == "sched" ]]; then
        out="results_agent_codex_sched_run${seed}"
        hot_start="results/seed_schedule.py"
        budget="$SCHED_BUDGET"
        sched_flag="--schedule-only"
    else
        out="results_agent_codex_fullopt_run${seed}"
        hot_start="results/seed_optimizer.py"
        budget="$FULL_BUDGET"
        sched_flag=""
    fi

    if [[ -f "$out/.done" ]]; then
        echo "[H] $out already done. Skipping."
        return
    fi
    mkdir -p "$out"
    echo "[H] $(date -u +%FT%TZ) Launching $mode seed $seed -> $out"

    pixi run python agent_cli.py \
        --provider codex \
        --model "$CODEX_MODEL" \
        $sched_flag \
        --wind-csv "$WIND_CSV" \
        --time-budget "$budget" \
        --hot-start "$hot_start" \
        --output-dir "$out" \
        --timeout-per-run 180 \
        > "$out/launch.log" 2>&1

    touch "$out/.done"
    echo "[H] $(date -u +%FT%TZ) Finished $mode seed $seed."
}

for mode in $MODES; do
    for seed in $SEEDS; do
        run_one "$mode" "$seed"
    done
done

echo "[H] $(date -u +%FT%TZ) All requested runs complete."
