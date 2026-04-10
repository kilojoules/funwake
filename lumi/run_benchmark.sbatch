#!/bin/bash
# Run the FunWake agent against a vLLM server on the same LUMI node.
#
# Usage:
#   # First, submit the vLLM server job:
#   MODEL_PRESET=qwen2.5-72b sbatch lumi/serve_vllm.sbatch
#   # Then submit this job with dependency:
#   MODEL_PRESET=qwen2.5-72b sbatch --dependency=after:<VLLM_JOBID> lumi/run_benchmark.sbatch
#
# Or set TIME_BUDGET (default 18000 = 5 hours):
#   MODEL_PRESET=qwen2.5-72b TIME_BUDGET=3600 sbatch --dependency=after:<JOBID> lumi/run_benchmark.sbatch

#SBATCH --job-name=funwake-agent
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=lumi/logs/agent_%j.log
#SBATCH --error=lumi/logs/agent_%j.err

set -e
mkdir -p lumi/logs

MODEL_PRESET=${MODEL_PRESET:-llama3.1-405b}
TIME_BUDGET=${TIME_BUDGET:-18000}
SCHEDULE_ONLY=${SCHEDULE_ONLY:-}

cd /scratch/project_465002609/julian/funwake

# Wait for vLLM server to be ready (up to 10 min)
echo "Waiting for vLLM server on localhost:8000..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "Server ready after $((i * 5))s"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "ERROR: vLLM server not ready after 10 min"
        exit 1
    fi
    sleep 5
done

# Get the served model ID
MODEL=$(curl -s http://localhost:8000/v1/models | python3 -c \
    "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
echo "Model: $MODEL"
echo "Preset: $MODEL_PRESET"
echo "Time budget: $((TIME_BUDGET / 60)) min"

OUTPUT_DIR="results_agent_${MODEL_PRESET//./_}"

# Build CLI args
CLI_ARGS=(
    --provider vllm
    --model "$MODEL"
    --base-url http://localhost:8000
    --wind-csv playground/pixwake/energy_island_10y_daily_av_wind.csv
    --time-budget "$TIME_BUDGET"
    --hot-start results/seed_optimizer.py
    --output-dir "$OUTPUT_DIR"
    --timeout-per-run 60
)

if [ -n "$SCHEDULE_ONLY" ]; then
    CLI_ARGS+=(--schedule-only --hot-start results/seed_schedule.py)
fi

echo "Running: python agent_cli.py ${CLI_ARGS[*]}"
python agent_cli.py "${CLI_ARGS[@]}"

echo "Agent finished. Results in $OUTPUT_DIR/"
