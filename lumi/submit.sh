#!/bin/bash
# Submit FunWake benchmark runs on LUMI.
#
# Each job runs vLLM + agent on the same node.
#
# Usage:
#   bash lumi/submit.sh gemma4-26b                  # single run, seed 1
#   bash lumi/submit.sh gemma4-26b 5                # 5 seeds
#   bash lumi/submit.sh gemma4-26b 5 3600           # 5 seeds, 1hr each
#   SCHEDULE_ONLY=1 bash lumi/submit.sh gemma4-26b 5 # schedule-only mode

set -e

MODEL_PRESET=${1:?Usage: bash lumi/submit.sh <model-preset> [num_seeds] [time_budget]}
NUM_SEEDS=${2:-1}
TIME_BUDGET=${3:-18000}
SCHEDULE_ONLY=${SCHEDULE_ONLY:-}

cd "$(dirname "$0")/.."
MODELS_JSON="$(pwd)/models.json"

if [ ! -f "$MODELS_JSON" ]; then
    echo "ERROR: models.json not found"
    exit 1
fi

NUM_GPUS=$(python3 -c "import json; print(json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['num_gpus'])")
PARTITION=$(python3 -c "import json; print(json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['lumi_partition'])")

MODEL_DIR="/scratch/project_465002609/models/${MODEL_PRESET}"
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model not found at $MODEL_DIR"
    echo "Download first: MODEL_PRESET=$MODEL_PRESET bash lumi/download_model.sh"
    exit 1
fi

echo "=== FunWake Benchmark Submission ==="
echo "Model:     $MODEL_PRESET"
echo "Partition: $PARTITION ($NUM_GPUS GPUs)"
echo "Seeds:     $NUM_SEEDS"
echo "Budget:    $((TIME_BUDGET / 60)) min per seed"
echo "Mode:      ${SCHEDULE_ONLY:+schedule-only}${SCHEDULE_ONLY:-full optimizer}"
echo ""

for SEED in $(seq 1 $NUM_SEEDS); do
    JOBID=$(MODEL_PRESET=$MODEL_PRESET TIME_BUDGET=$TIME_BUDGET SEED=$SEED SCHEDULE_ONLY=$SCHEDULE_ONLY \
        sbatch \
        --partition="$PARTITION" \
        --gpus-per-node="$NUM_GPUS" \
        --parsable \
        lumi/run_benchmark.sbatch)
    echo "Seed $SEED: job $JOBID (output: results_agent_${MODEL_PRESET//./_}_s${SEED}/)"
done

echo ""
echo "Monitor: squeue -u $(whoami)"
echo "Logs:    ls lumi/logs/funwake_*.log"
