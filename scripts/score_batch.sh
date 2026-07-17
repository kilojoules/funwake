#!/bin/bash
# Score all optimizers in batch

export JAX_ENABLE_X64=True
export PYTHONPATH=playground/pixwake/src:$PYTHONPATH

LOG=results_agent_claude_fullopt/attempt_log.json

for script in results_agent_claude_fullopt/iter_*.py; do
    echo "Scoring $script..."
    timeout 90 python tools/run_optimizer.py "$script" --timeout 60 --log "$LOG" 2>&1 | tail -20
    echo "---"
done

echo "Final results:"
python tools/get_status.py --log "$LOG" 2>&1 || cat "$LOG"
