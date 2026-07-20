#!/bin/bash
# Replicated schedule-search reruns for the reviewer's Major #2:
#   - 3 seeds per agent (n=1 -> n=3, converts Fig 2 to a distribution)
#   - explicit --model pin (model identity logged, not account-default)
#   - CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 (memory channel provably off)
#   - CLI versions logged; per-iteration artifacts + session transcript retained
#
# Runs 6 x 5h agent loops. Long. Run in a persistent terminal (tmux/screen),
# NOT inside a tool harness that may kill long jobs.
#
# Usage:  bash run_reruns.sh            # all 6 sequentially (~30h)
#         SEEDS="1" bash run_reruns.sh  # one seed each (quick smoke: still 5h each)
set -uo pipefail
cd "$(dirname "$0")/.."

# ---- pinned models (edit if you want a different tier; identity gets logged) ----
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-4-5-20251101}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3-flash-preview}"
WIND_CSV="${WIND_CSV:-$HOME/clusters/energy_island_10y_daily_av_wind.csv}"
BUDGET="${BUDGET:-18000}"            # 5 hours
SEEDS="${SEEDS:-1 2 3}"
PP=dependencies/pixwake/src

run_one () {
  local provider="$1" model="$2" tag="$3" seed="$4"
  local out="results_rerun_${tag}_seed${seed}"
  mkdir -p "$out"
  echo "=== $(date) | $tag seed $seed | model=$model | budget=${BUDGET}s ==="
  # log CLI version + pinned model up front
  { echo "provider=$provider"; echo "model=$model"; echo "seed=$seed";
    echo "budget_s=$BUDGET"; echo -n "cli_version="; \
    if [ "$provider" = "claude-code" ]; then claude --version; else gemini --version; fi;
  } > "$out/run_meta.txt" 2>&1
  PYTHONPATH="$PP" pixi run python agent_cli.py \
      --provider "$provider" --model "$model" \
      --schedule-only --disable-auto-memory \
      --time-budget "$BUDGET" \
      --wind-csv "$WIND_CSV" \
      --hot-start results/seed_schedule.py \
      --output-dir "$out" 2>&1 | tee "$out/run.log"
  # best-effort: retain the newest Claude session transcript
  if [ "$provider" = "claude-code" ]; then
    latest=$(ls -t "$HOME/.claude/projects/"*/*.jsonl 2>/dev/null | head -1)
    [ -n "$latest" ] && cp "$latest" "$out/session_transcript.jsonl" 2>/dev/null || true
  fi
}

for s in $SEEDS; do run_one claude-code "$CLAUDE_MODEL" claude "$s"; done
for s in $SEEDS; do run_one gemini-cli  "$GEMINI_MODEL" gemini "$s"; done
echo "=== all reruns done: $(date) ==="
