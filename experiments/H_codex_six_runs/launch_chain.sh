#!/usr/bin/env bash
# Experiment H: 6 sequential codex runs (3 sched + 3 full).
#
# Sequential because Codex CLI single-account concurrency is 1.
# Total wall-clock estimate: 3 * 3hr + 3 * 4.5hr = ~22.5 hr.
#
# Cross-run isolation:
#   Codex's workspace-write sandbox lets the agent read every directory
#   in the project. We observed run 2 transferring schedules verbatim
#   from run 1. To keep replicates iid, we hide the other completed
#   codex result dirs (by renaming to .hidden_<dir>/) for the duration
#   of each new run, then restore them after the run finishes (or on
#   error / Ctrl-C, via trap).
#
# Resume-safe: each run writes .done on success.
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

HIDE_PREFIX=".hidden_"
declare -a HIDDEN=()

restore_hidden() {
    for d in "${HIDDEN[@]:-}"; do
        [[ -z "$d" ]] && continue
        if [[ -d "${HIDE_PREFIX}${d}" ]]; then
            mv "${HIDE_PREFIX}${d}" "$d"
            echo "[H]   restored $d"
        fi
    done
    HIDDEN=()
}
trap restore_hidden EXIT INT TERM

hide_siblings() {
    # Hide every results_agent_*/ (codex, claude, gemini, deepseek, ...)
    # except the active run dir. This ensures cross-run isolation —
    # the agent cannot read prior agents' iter scripts.
    local active="$1"
    HIDDEN=()
    for d in results_agent_*/; do
        d="${d%/}"
        [[ -z "$d" || "$d" == "$active" ]] && continue
        if [[ -d "$d" ]]; then
            mv "$d" "${HIDE_PREFIX}${d}"
            HIDDEN+=("$d")
        fi
    done
    if [[ ${#HIDDEN[@]} -gt 0 ]]; then
        echo "[H]   hid ${#HIDDEN[@]} sibling dirs"
    fi
}

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
    hide_siblings "$out"

    echo "[H] $(date -u +%FT%TZ) Launching $mode seed $seed -> $out"

    set +e
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
    rc=$?
    set -e

    restore_hidden
    if [[ $rc -ne 0 ]]; then
        echo "[H] $(date -u +%FT%TZ) FAILED $mode seed $seed (rc=$rc). Check $out/launch.log"
        return $rc
    fi

    touch "$out/.done"
    echo "[H] $(date -u +%FT%TZ) Finished $mode seed $seed."
}

if [[ -n "${PAIRS:-}" ]]; then
    # PAIRS="sched:2 sched:3 full:1 full:2 full:3"
    for pair in $PAIRS; do
        run_one "${pair%%:*}" "${pair##*:}"
    done
else
    for mode in $MODES; do
        for seed in $SEEDS; do
            run_one "$mode" "$seed"
        done
    done
fi

echo "[H] $(date -u +%FT%TZ) All requested runs complete."
