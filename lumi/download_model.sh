#!/bin/bash
# Download Llama 3.1 405B AWQ INT4 to LUMI scratch.
# Run from a login node (has internet access).
#
# Usage: bash lumi/download_model.sh

set -e

MODEL_DIR=/scratch/project_465002609/models/llama-405b-awq

echo "Downloading Llama 3.1 405B AWQ INT4 to ${MODEL_DIR}..."
echo "This will take 1-2 hours (~200 GB)."

pip install --user huggingface-hub 2>/dev/null

huggingface-cli download \
    huggingface/Meta-Llama-3.1-405B-Instruct-AWQ-INT4 \
    --local-dir ${MODEL_DIR}

echo "Done. Model at ${MODEL_DIR}"
ls -lh ${MODEL_DIR}
