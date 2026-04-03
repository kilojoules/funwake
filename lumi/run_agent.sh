#!/bin/bash
# Run the FunWake agent against the vLLM server on the same LUMI node.
#
# Usage (from the compute node where vLLM is running):
#   bash lumi/run_agent.sh [time_budget_seconds]
#
# Or submit as a second job that depends on the vLLM server job:
#   sbatch --dependency=after:JOBID lumi/run_agent.sbatch

set -e

TIME_BUDGET=${1:-18000}  # default 5 hours

cd /scratch/project_465002609/julian/funwake

# Wait for vLLM server to be ready
echo "Waiting for vLLM server on localhost:8000..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 5
done

# Check what model is served
MODEL=$(curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
echo "Model: $MODEL"

# Run the agent
python agent_cli.py \
    --provider vllm \
    --model "$MODEL" \
    --base-url http://localhost:8000 \
    --wind-csv /scratch/project_465002609/julian/funwake/playground/pixwake/energy_island_10y_daily_av_wind.csv \
    --time-budget $TIME_BUDGET \
    --hot-start results/seed_optimizer.py \
    --output-dir results_agent_llama405b \
    --timeout-per-run 60

echo "Agent finished. Results in results_agent_llama405b/"
