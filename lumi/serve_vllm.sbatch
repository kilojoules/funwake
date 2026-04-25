#!/bin/bash
# Serve an open-source model via vLLM on LUMI.
#
# Usage:
#   MODEL_PRESET=qwen2.5-72b sbatch lumi/serve_vllm.sbatch
#   MODEL_PRESET=llama3.1-405b sbatch lumi/serve_vllm.sbatch
#
# The MODEL_PRESET must match a key in models.json.
# GPU count and partition are determined by the preset.

#SBATCH --job-name=vllm-serve
#SBATCH --account=project_465002609
#SBATCH --time=06:00:00
#SBATCH --output=lumi/logs/vllm_%j.log
#SBATCH --error=lumi/logs/vllm_%j.err

set -e
mkdir -p lumi/logs

MODEL_PRESET=${MODEL_PRESET:-llama3.1-405b}
PROJECT_ROOT=$(pwd)
MODELS_JSON="${PROJECT_ROOT}/models.json"

if [ ! -f "$MODELS_JSON" ]; then
    echo "ERROR: models.json not found at $MODELS_JSON"
    exit 1
fi

# Read model config from models.json
HF_ID=$(python3 -c "import json; print(json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['hf_id'])")
NUM_GPUS=$(python3 -c "import json; print(json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['num_gpus'])")
MAX_MODEL_LEN=$(python3 -c "import json; print(json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['vllm_args']['max-model-len'])")
DTYPE=$(python3 -c "import json; print(json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['vllm_args']['dtype'])")
QUANT=$(python3 -c "import json; q=json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['vllm_args'].get('quantization'); print(q if q else '')")

MODEL_DIR="/scratch/project_465002609/models/${MODEL_PRESET}"

echo "Model preset: $MODEL_PRESET"
echo "HF ID: $HF_ID"
echo "Model dir: $MODEL_DIR"
echo "GPUs: $NUM_GPUS"
echo "Max model len: $MAX_MODEL_LEN"
echo "Dtype: $DTYPE"
echo "Quantization: ${QUANT:-none}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model not found at $MODEL_DIR"
    echo "Run: MODEL_PRESET=$MODEL_PRESET bash lumi/download_model.sh"
    exit 1
fi

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Find the latest container
export SIF=$(ls -t /appl/local/laifs/containers/lumi-multitorch-*/lumi-multitorch-full-*.sif 2>/dev/null | head -1)
if [ -z "$SIF" ]; then
    echo "ERROR: No LAIF container found. Trying EasyBuild module..."
    module load LUMI/24.03 partition/G
    module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250410
    export SIF=$SINGULARITY_IMAGE
fi
echo "Using container: $SIF"

# RCCL settings for multi-GCD communication
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3

# Build vLLM args
VLLM_ARGS="--model $MODEL_DIR --tensor-parallel-size $NUM_GPUS --dtype $DTYPE --max-model-len $MAX_MODEL_LEN --port 8000 --host 0.0.0.0"
if [ -n "$QUANT" ]; then
    VLLM_ARGS="$VLLM_ARGS --quantization $QUANT"
fi

echo "Starting vLLM: $VLLM_ARGS"

singularity exec $SIF \
    python -m vllm.entrypoints.openai.api_server $VLLM_ARGS
