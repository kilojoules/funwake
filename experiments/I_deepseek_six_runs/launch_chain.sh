#!/usr/bin/env bash
# DeepSeek 6-run chain (3 sched + 3 full) via the vLLM/OpenAI-compatible
# runner pointed at api.deepseek.com.
#
# Mirrors experiments/H_codex_six_runs/launch_chain.sh: sibling-hiding,
# resume on .done, optional PAIRS env for ordered subsets.
#
# Pre-launch:
#   - ~/.deepsk holds the API key (one line, sk-...).
#   - DEEPSEEK_MODEL defaults to deepseek-v4-flash; override for v4-pro.

set -euo pipefail
cd "$(dirname "$0")/../.."

API_KEY_FILE=${API_KEY_FILE:-$HOME/.deepsk}
[[ -f "$API_KEY_FILE" ]] || { echo "[I] API key file $API_KEY_FILE not found"; exit 2; }
API_KEY=$(cat "$API_KEY_FILE" | tr -d '[:space:]')

DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-deepseek-v4-flash}
BASE_URL=${BASE_URL:-https://api.deepseek.com}
MODES=${MODES:-"sched full"}
SEEDS=${SEEDS:-"1 2 3"}
WIND_CSV=${WIND_CSV:-~/clusters/energy_island_10y_daily_av_wind.csv}
SCHED_BUDGET=${SCHED_BUDGET:-10800}    # 3 hr
FULL_BUDGET=${FULL_BUDGET:-16200}      # 4.5 hr

HIDE_PREFIX=".hidden_"
declare -a HIDDEN=()

restore_hidden() {
    for d in "${HIDDEN[@]:-}"; do
        [[ -z "$d" ]] && continue
        if [[ -d "${HIDE_PREFIX}${d}" ]]; then
            mv "${HIDE_PREFIX}${d}" "$d"
        fi
    done
    HIDDEN=()
}
trap restore_hidden EXIT INT TERM

hide_siblings() {
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
        echo "[I]   hid ${#HIDDEN[@]} sibling dirs"
    fi
}

run_one() {
    local mode="$1"
    local seed="$2"

    local out hot_start budget sched_flag
    if [[ "$mode" == "sched" ]]; then
        out="results_agent_deepseek_sched_run${seed}"
        hot_start="results/seed_schedule.py"
        budget="$SCHED_BUDGET"
        sched_flag="--schedule-only"
    else
        out="results_agent_deepseek_fullopt_run${seed}"
        hot_start="results/seed_optimizer.py"
        budget="$FULL_BUDGET"
        sched_flag=""
    fi

    if [[ -f "$out/.done" ]]; then
        echo "[I] $out already done. Skipping."
        return
    fi

    mkdir -p "$out"
    hide_siblings "$out"

    echo "[I] $(date -u +%FT%TZ) Launching $mode seed $seed -> $out (model=$DEEPSEEK_MODEL)"

    set +e
    DEEPSEEK_API_KEY="$API_KEY" pixi run python agent_cli.py \
        --provider vllm \
        --model "$DEEPSEEK_MODEL" \
        --base-url "$BASE_URL" \
        --api-key "$API_KEY" \
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
        echo "[I] $(date -u +%FT%TZ) FAILED $mode seed $seed (rc=$rc). Check $out/launch.log"
        return $rc
    fi

    touch "$out/.done"
    echo "[I] $(date -u +%FT%TZ) Finished $mode seed $seed."
}

if [[ -n "${PAIRS:-}" ]]; then
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

echo "[I] $(date -u +%FT%TZ) All requested runs complete."
