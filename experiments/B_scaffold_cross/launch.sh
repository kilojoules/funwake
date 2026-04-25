#!/usr/bin/env bash
# Experiment B: scaffold-cross. Run Claude under the structured-output
# scaffold (no `claude -p` outer loop) on both interfaces.
#
# REQUIRES: runners/anthropic_api_runner.py (B-a path). If absent, this
# script exits with a TODO message.
#
# Pre-launch checklist:
#   - ANTHROPIC_API_KEY set, with a budget that allows ~5hr × 2 of
#     direct API calls (rough estimate: ~$50–150 depending on tool-use
#     frequency).
#   - The anthropic-api runner has been smoke-tested in a short loop
#     before being committed to a 5 hr run.
#   - Output dirs do not collide with prior runs.
#
# Usage:
#   bash experiments/B_scaffold_cross/launch.sh sched     # only B1
#   bash experiments/B_scaffold_cross/launch.sh fullopt   # only B2
#   bash experiments/B_scaffold_cross/launch.sh both      # default

set -euo pipefail
cd "$(dirname "$0")/../.."

MODE=${1:-both}
TIME_BUDGET=${TIME_BUDGET:-18000}
WIND_CSV=${WIND_CSV:-~/clusters/energy_island_10y_daily_av_wind.csv}
ANTHROPIC_MODEL=${ANTHROPIC_MODEL:-claude-sonnet-4-6}

if [[ ! -f runners/anthropic_api_runner.py ]]; then
    cat <<MSG
[B] runners/anthropic_api_runner.py is missing.
[B] Implement the structured-output Anthropic-API runner before launching:
[B]   - Subclass BaseRunner (see runners/vllm_runner.py for the pattern).
[B]   - Call the Anthropic Messages API with tool_use blocks for
[B]     read_file, write_file, run_tests, run_optimizer.
[B]   - Persist memory across turns the same way ClaudeCodeRunner does.
[B]   - Smoke test with --time-budget 600 first.
[B] Aborting.
MSG
    exit 2
fi

run_b1_sched() {
    local out=results_agent_claude_anthropic_api_sched
    [[ -f "$out/.done" ]] && { echo "[B1] Already complete."; return; }
    mkdir -p "$out"
    pixi run python agent_cli.py \
        --provider anthropic-api \
        --model "$ANTHROPIC_MODEL" \
        --schedule-only \
        --wind-csv "$WIND_CSV" \
        --time-budget "$TIME_BUDGET" \
        --hot-start results/seed_schedule.py \
        --output-dir "$out" \
        --timeout-per-run 180 \
        > "$out/launch.log" 2>&1
    touch "$out/.done"
}

run_b2_fullopt() {
    local out=results_agent_claude_anthropic_api_fullopt
    [[ -f "$out/.done" ]] && { echo "[B2] Already complete."; return; }
    mkdir -p "$out"
    pixi run python agent_cli.py \
        --provider anthropic-api \
        --model "$ANTHROPIC_MODEL" \
        --wind-csv "$WIND_CSV" \
        --time-budget "$TIME_BUDGET" \
        --hot-start results/seed_optimizer.py \
        --output-dir "$out" \
        --timeout-per-run 180 \
        > "$out/launch.log" 2>&1
    touch "$out/.done"
}

case "$MODE" in
    sched)   run_b1_sched ;;
    fullopt) run_b2_fullopt ;;
    both)    run_b1_sched; run_b2_fullopt ;;
    *) echo "[B] Unknown mode: $MODE"; exit 2 ;;
esac

echo "[B] done."
