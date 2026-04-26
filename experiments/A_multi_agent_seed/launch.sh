#!/usr/bin/env bash
# Experiment A: 4 additional schedule-only agent runs each for Claude Code
# and Gemini CLI. Existing run 1 is the original 5hr loop; we extend to n=5.
#
# Pre-launch checklist:
#   - ANTHROPIC_API_KEY set? Verify quota for ~5 hr × 5 runs of Claude usage.
#   - GEMINI_API_KEY set? Same for Gemini.
#   - --hot-start file exists at results/seed_schedule.py
#   - Existing results_agent_*_run1/ dirs are NOT in the loop (we resume from run2).
#
# Usage:
#   bash experiments/A_multi_agent_seed/launch.sh           # both agents, runs 2..5
#   AGENTS=claude bash experiments/A_multi_agent_seed/launch.sh   # only Claude
#   RUNS="2 3" bash experiments/A_multi_agent_seed/launch.sh      # subset
#   PARALLEL=2 bash experiments/A_multi_agent_seed/launch.sh      # 2 concurrent
#
# Output: results_agent_<agent>_sched_run<N>/

set -euo pipefail
cd "$(dirname "$0")/../.."

AGENTS=${AGENTS:-"claude gemini codex"}
RUNS=${RUNS:-"2 3 4 5"}
PARALLEL=${PARALLEL:-1}
# Schedule-only saturates around iter 150. Empirical 5hr run hit 316 iter.
# Use formula: t = 2 * 150 * 30s ~= 9000s. Allow 3 hr (10800s) for margin.
TIME_BUDGET=${TIME_BUDGET:-10800}     # 3 hr
WIND_CSV=${WIND_CSV:-~/clusters/energy_island_10y_daily_av_wind.csv}
HOT_START=${HOT_START:-results/seed_schedule.py}
CODEX_MODEL=${CODEX_MODEL:-gpt-5.5}

provider_for() {
    case "$1" in
        claude) echo "claude-code" ;;
        gemini) echo "gemini-cli" ;;
        codex)  echo "codex" ;;
        *) echo "unknown"; return 1 ;;
    esac
}

run_one() {
    local agent="$1"
    local run="$2"
    local provider; provider=$(provider_for "$agent")
    local out_dir="results_agent_${agent}_sched_run${run}"

    if [[ -f "$out_dir/.done" ]]; then
        echo "[A] $out_dir already complete. Skipping."
        return
    fi
    mkdir -p "$out_dir"

    local log="$out_dir/launch.log"
    echo "[A] Launching $agent run $run -> $out_dir"

    local provider_args=(--provider "$provider")
    if [[ "$provider" == "codex" ]]; then
        provider_args+=(--model "$CODEX_MODEL")
    fi

    pixi run python agent_cli.py \
        "${provider_args[@]}" \
        --schedule-only \
        --wind-csv "$WIND_CSV" \
        --time-budget "$TIME_BUDGET" \
        --hot-start "$HOT_START" \
        --output-dir "$out_dir" \
        --timeout-per-run 180 \
        > "$log" 2>&1

    touch "$out_dir/.done"
    echo "[A] $agent run $run done."
}

export -f run_one provider_for

JOBS=()
for agent in $AGENTS; do
    for run in $RUNS; do
        JOBS+=("$agent $run")
    done
done

echo "[A] ${#JOBS[@]} runs queued (parallel=$PARALLEL)."

if [[ "$PARALLEL" -le 1 ]]; then
    for j in "${JOBS[@]}"; do
        # shellcheck disable=SC2086
        run_one $j
    done
else
    printf '%s\n' "${JOBS[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'run_one $1 $2' _ {}
fi

echo "[A] All requested runs finished."
