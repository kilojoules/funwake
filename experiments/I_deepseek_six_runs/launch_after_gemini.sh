#!/usr/bin/env bash
# Wait for the running gemini sched chain to finish, then launch the
# deepseek 6-run chain. Avoids hide_siblings collisions between two
# concurrent renaming chains.
#
# Usage:
#   GEMINI_PID=58892 nohup bash \
#       experiments/I_deepseek_six_runs/launch_after_gemini.sh \
#       > experiments/I_deepseek_six_runs/chain.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

GEMINI_PID=${GEMINI_PID:?"set GEMINI_PID to the active gemini chain PID"}

echo "[ds-wait] $(date -u +%FT%TZ) Waiting for gemini chain PID $GEMINI_PID..."
while kill -0 "$GEMINI_PID" 2>/dev/null; do
    sleep 60
done
echo "[ds-wait] $(date -u +%FT%TZ) Gemini chain exited. Launching deepseek chain."

bash experiments/I_deepseek_six_runs/launch_chain.sh
echo "[ds-wait] $(date -u +%FT%TZ) Deepseek chain done."
