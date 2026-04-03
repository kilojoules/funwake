#!/bin/bash
#SBATCH --job-name=vllm-405b
#SBATCH --account=project_465002609
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=06:00:00
#SBATCH --output=lumi/logs/vllm_%j.log
#SBATCH --error=lumi/logs/vllm_%j.err

# Serve Llama 3.1 405B via vLLM on a full LUMI node (8 GCDs).
# The agent connects to http://localhost:8000 from the same node.

set -e
mkdir -p lumi/logs

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

MODEL=/scratch/project_465002609/models/llama-405b-awq

echo "Starting vLLM server with model: $MODEL"
echo "Tensor parallel: 8 GCDs"
echo "Port: 8000"

singularity exec $SIF \
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --tensor-parallel-size 8 \
        --quantization awq \
        --dtype float16 \
        --max-model-len 8192 \
        --port 8000 \
        --host 0.0.0.0
