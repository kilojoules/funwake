#!/usr/bin/env bash
# Generic multi-agent schedule-only chain launcher with sibling hiding.
# Mirrors experiments/H_codex_six_runs/launch_chain.sh but parameterized.
#
# Pre-launch:
#   - The corresponding agent CLI is logged in / quota available.
#   - Run 1 already exists for the chosen agent (or set FROM_RUN=1).
#
# Usage:
#   AGENT=gemini RUNS="2 3 4" bash experiments/A_multi_agent_seed/launch_chain.sh
#   AGENT=claude RUNS="2 3 4 5" bash experiments/A_multi_agent_seed/launch_chain.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

AGENT=${AGENT:?"set AGENT to claude|gemini|codex"}
RUNS=${RUNS:-"2 3 4"}
WIND_CSV=${WIND_CSV:-~/clusters/energy_island_10y_daily_av_wind.csv}
HOT_START=${HOT_START:-results/seed_schedule.py}
TIME_BUDGET=${TIME_BUDGET:-10800}   # 3 hr default for sched
CODEX_MODEL=${CODEX_MODEL:-gpt-5.5}

case "$AGENT" in
    claude) PROVIDER=claude-code ;;
    gemini) PROVIDER=gemini-cli ;;
    codex)  PROVIDER=codex ;;
    *) echo "[A] unknown AGENT=$AGENT"; exit 2 ;;
esac

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
        echo "[A]   hid ${#HIDDEN[@]} sibling dirs"
    fi
}

run_one() {
    local n="$1"
    local out="results_agent_${AGENT}_sched_run${n}"
    if [[ -f "$out/.done" ]]; then
        echo "[A] $out done. Skipping."
        return
    fi
    mkdir -p "$out"
    hide_siblings "$out"

    # Build optional --model arg as a string; bash 3.2 (macOS) errors on
    # ${arr[@]} expansion of empty arrays under set -u, so use a flat string.
    local model_arg=""
    if [[ "$PROVIDER" == "codex" ]]; then
        model_arg="--model $CODEX_MODEL"
    fi

    echo "[A] $(date -u +%FT%TZ) Launching $AGENT seed $n -> $out"

    set +e
    # shellcheck disable=SC2086
    pixi run python agent_cli.py \
        --provider "$PROVIDER" \
        $model_arg \
        --schedule-only \
        --wind-csv "$WIND_CSV" \
        --time-budget "$TIME_BUDGET" \
        --hot-start "$HOT_START" \
        --output-dir "$out" \
        --timeout-per-run 180 \
        > "$out/launch.log" 2>&1
    rc=$?
    set -e

    restore_hidden
    if [[ $rc -ne 0 ]]; then
        echo "[A] $(date -u +%FT%TZ) FAILED $AGENT seed $n (rc=$rc). See $out/launch.log"
        return $rc
    fi
    touch "$out/.done"
    echo "[A] $(date -u +%FT%TZ) Finished $AGENT seed $n."
}

for n in $RUNS; do
    run_one "$n"
done

echo "[A] $(date -u +%FT%TZ) All requested runs complete."
