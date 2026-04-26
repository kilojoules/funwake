#!/usr/bin/env bash
# Wait for the codex chain (H) to finish, then launch the Gemini sched
# chain. Avoids contention between H's hide_siblings and A's
# hide_siblings (both rename results_agent_*/).
#
# Matrix eval (no dir renames) is safe to keep running alongside.
#
# Usage:
#   CODEX_PID=18467 nohup bash \
#       experiments/A_multi_agent_seed/launch_gemini_after_codex.sh \
#       > experiments/A_multi_agent_seed/gemini_chain.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

CODEX_PID=${CODEX_PID:?"set CODEX_PID to the launch_chain.sh PID"}
GEMINI_RUNS=${GEMINI_RUNS:-"2 3 4"}

echo "[gem-wait] $(date -u +%FT%TZ) Waiting for codex chain PID $CODEX_PID..."
while kill -0 "$CODEX_PID" 2>/dev/null; do
    sleep 60
done
echo "[gem-wait] $(date -u +%FT%TZ) Codex chain exited. Launching gemini chain."

# Confirm the gemini run-1 dir exists; aggregate.py needs it.
if [[ ! -d results_agent_gemini_cli_5hr && ! -d .hidden_results_agent_gemini_cli_5hr ]]; then
    echo "[gem-wait] WARNING: results_agent_gemini_cli_5hr not found in either visible or .hidden_ form."
fi

AGENT=gemini RUNS="$GEMINI_RUNS" bash experiments/A_multi_agent_seed/launch_chain.sh

echo "[gem-wait] $(date -u +%FT%TZ) Gemini chain done."
