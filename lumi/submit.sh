#!/bin/bash
# Submit a vLLM server + agent benchmark run on LUMI.
#
# Reads models.json to determine partition, GPU count, etc.
# Submits two chained SLURM jobs: (1) vLLM server, (2) agent.
#
# Usage:
#   bash lumi/submit.sh qwen2.5-coder-32b          # cheapest, 1 GCD
#   bash lumi/submit.sh qwen2.5-72b                 # 1x MI250X GCD
#   bash lumi/submit.sh llama3.3-70b                # 1x MI250X GCD
#   bash lumi/submit.sh llama3.1-405b               # 8x GCDs (full node)
#   bash lumi/submit.sh llama3.1-405b 3600          # custom time budget (1hr)
#   SCHEDULE_ONLY=1 bash lumi/submit.sh qwen2.5-72b # schedule-only mode

set -e

MODEL_PRESET=${1:?Usage: bash lumi/submit.sh <model-preset> [time_budget_seconds]}
TIME_BUDGET=${2:-18000}
SCHEDULE_ONLY=${SCHEDULE_ONLY:-}

cd "$(dirname "$0")/.."
MODELS_JSON="$(pwd)/models.json"

if [ ! -f "$MODELS_JSON" ]; then
    echo "ERROR: models.json not found"
    exit 1
fi

# Read model config
NUM_GPUS=$(python3 -c "import json; print(json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['num_gpus'])")
PARTITION=$(python3 -c "import json; print(json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['lumi_partition'])")

echo "=== FunWake Benchmark Submission ==="
echo "Model:     $MODEL_PRESET"
echo "Partition: $PARTITION"
echo "GPUs:      $NUM_GPUS"
echo "Budget:    $((TIME_BUDGET / 60)) min"
echo "Schedule:  ${SCHEDULE_ONLY:-full optimizer}"
echo ""

# Check model is downloaded
MODEL_DIR="/scratch/project_465002609/models/${MODEL_PRESET}"
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model not found at $MODEL_DIR"
    echo "Download first: MODEL_PRESET=$MODEL_PRESET bash lumi/download_model.sh"
    exit 1
fi

# Submit vLLM server job (override partition + GPUs from models.json)
VLLM_JOBID=$(MODEL_PRESET=$MODEL_PRESET sbatch \
    --partition="$PARTITION" \
    --gpus-per-node="$NUM_GPUS" \
    --parsable \
    lumi/serve_vllm.sbatch)
echo "Submitted vLLM server: job $VLLM_JOBID (partition=$PARTITION, gpus=$NUM_GPUS)"

# Submit agent job chained after vLLM server
# Agent runs on small-g (1 GCD) — it only needs CPU for the optimizer scoring
AGENT_JOBID=$(MODEL_PRESET=$MODEL_PRESET TIME_BUDGET=$TIME_BUDGET SCHEDULE_ONLY=$SCHEDULE_ONLY \
    sbatch \
    --dependency=after:"$VLLM_JOBID" \
    --parsable \
    lumi/run_benchmark.sbatch)
echo "Submitted agent:       job $AGENT_JOBID (depends on $VLLM_JOBID)"

echo ""
echo "Monitor: squeue -u $(whoami)"
echo "Logs:    tail -f lumi/logs/vllm_${VLLM_JOBID}.log"
echo "         tail -f lumi/logs/agent_${AGENT_JOBID}.log"
