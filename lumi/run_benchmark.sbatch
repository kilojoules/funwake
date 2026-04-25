#!/bin/bash
# Run a complete FunWake benchmark: vLLM server + agent on the same node.
#
# Usage:
#   MODEL_PRESET=gemma4-26b SEED=1 sbatch lumi/run_benchmark.sbatch
#   MODEL_PRESET=gemma4-26b SEED=2 sbatch lumi/run_benchmark.sbatch
#
# The vLLM server runs in the background on the node's GPU(s).
# The agent connects to localhost:8000 and runs the optimization loop.

#SBATCH --job-name=funwake
#SBATCH --account=project_465002609
#SBATCH --time=07:00:00
#SBATCH --output=lumi/logs/funwake_%j.log
#SBATCH --error=lumi/logs/funwake_%j.err
# Partition and GPUs are overridden by submit.sh based on model size

set -e

MODEL_PRESET=${MODEL_PRESET:-gemma4-26b}
TIME_BUDGET=${TIME_BUDGET:-18000}
SEED=${SEED:-1}
SCHEDULE_ONLY=${SCHEDULE_ONLY:-}
# Hot-start script the agent begins from. Default matches the
# paper's original behavior; overridden by the iterative seed
# driver to point at the previous generation's best script.
HOT_START=${HOT_START:-results/seed_optimizer.py}

cd /scratch/project_465002609/julian/funwake
mkdir -p lumi/logs

MODELS_JSON="$(pwd)/models.json"
HF_ID=$(pixi run python -c "import json; print(json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['hf_id'])")
NUM_GPUS=$(pixi run python -c "import json; print(json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['num_gpus'])")
MAX_MODEL_LEN=$(pixi run python -c "import json; print(json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['vllm_args']['max-model-len'])")
DTYPE=$(pixi run python -c "import json; print(json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['vllm_args']['dtype'])")
QUANT=$(pixi run python -c "import json; q=json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['vllm_args'].get('quantization'); print(q if q else '')")
REASONING=$(pixi run python -c "import json; m=json.load(open('$MODELS_JSON'))['$MODEL_PRESET']; print('1' if m.get('reasoning') else '')")
REASONING_PARSER=$(pixi run python -c "import json; m=json.load(open('$MODELS_JSON'))['$MODEL_PRESET']; print(m.get('reasoning_parser', 'deepseek_r1'))")
MODEL_DIR="/scratch/project_465002609/models/${MODEL_PRESET}"

echo "=== FunWake Benchmark ==="
echo "Model:  $MODEL_PRESET ($HF_ID)"
echo "GPUs:   $NUM_GPUS"
echo "Seed:   $SEED"
echo "Budget: $((TIME_BUDGET / 60)) min"
echo ""

# ── Start vLLM server in background ──────────────────────────────────

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF=$(ls -t /appl/local/laifs/containers/lumi-multitorch-*/lumi-multitorch-full-*.sif 2>/dev/null | head -1)
if [ -z "$SIF" ]; then
    module load LUMI/24.03 partition/G
    module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250410
    export SIF=$SINGULARITY_IMAGE
fi
echo "Container: $SIF"

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3

# AMD GPU visibility: LUMI uses ROCR but vLLM/Ray expects HIP
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
fi

VLLM_ARGS="--model $MODEL_DIR --tensor-parallel-size $NUM_GPUS --dtype $DTYPE --max-model-len $MAX_MODEL_LEN --port 8000 --host 0.0.0.0"
if [ -n "$QUANT" ]; then
    VLLM_ARGS="$VLLM_ARGS --quantization $QUANT"
fi
if [ -n "$REASONING" ]; then
    VLLM_ARGS="$VLLM_ARGS --reasoning-parser $REASONING_PARSER"
    echo "Reasoning mode enabled (parser: $REASONING_PARSER)"
fi

VLLM_LOG="$(pwd)/lumi/logs/vllm_${SLURM_JOB_ID}.log"
echo "Starting vLLM: $VLLM_ARGS"
echo "vLLM log: $VLLM_LOG"
singularity exec $SIF \
    python -m vllm.entrypoints.openai.api_server $VLLM_ARGS \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Cleanup on exit
trap "echo 'Stopping vLLM...'; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null" EXIT

# ── Wait for vLLM to be ready ────────────────────────────────────────

echo "Waiting for vLLM server..."
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

MODEL=$(curl -s http://localhost:8000/v1/models | python3 -c \
    "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
echo "Serving: $MODEL"

# ── Run the agent ────────────────────────────────────────────────────

EXP_TAG=${EXP_TAG:-}  # optional suffix to distinguish scaffolding versions
TAG_SUFFIX=""
if [ -n "$EXP_TAG" ]; then
    TAG_SUFFIX="_${EXP_TAG}"
fi

if [ -n "$SCHEDULE_ONLY" ]; then
    OUTPUT_DIR="results_agent_${MODEL_PRESET//./_}${TAG_SUFFIX}_sched_s${SEED}"
else
    OUTPUT_DIR="results_agent_${MODEL_PRESET//./_}${TAG_SUFFIX}_s${SEED}"
fi

# Provider: set via PROVIDER env var, default to vllm
PROVIDER=${PROVIDER:-vllm}

# Unbuffered Python output so we can see agent turns in the log
export PYTHONUNBUFFERED=1

CLI_ARGS=(
    --provider "$PROVIDER"
    --model "$MODEL"
    --base-url http://localhost:8000
    --wind-csv playground/pixwake/energy_island_10y_daily_av_wind.csv
    --time-budget "$TIME_BUDGET"
    --hot-start "$HOT_START"
    --output-dir "$OUTPUT_DIR"
    --timeout-per-run 60
)

if [ -n "$SCHEDULE_ONLY" ]; then
    CLI_ARGS+=(--schedule-only)
    # Default to schedule-only seed if caller didn't override HOT_START
    if [ "$HOT_START" = "results/seed_optimizer.py" ]; then
        CLI_ARGS+=(--hot-start results/seed_schedule.py)
    fi
fi

echo ""
echo "Running: pixi run python agent_cli.py ${CLI_ARGS[*]}"
pixi run python agent_cli.py "${CLI_ARGS[@]}"

echo ""
echo "Agent finished. Results in $OUTPUT_DIR/"
