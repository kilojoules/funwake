#!/bin/bash
# Download a model from HuggingFace to LUMI scratch.
# Run from a login node (has internet access).
#
# Usage:
#   MODEL_PRESET=gemma4-26b bash lumi/download_model.sh
#   MODEL_PRESET=qwen2.5-72b bash lumi/download_model.sh
#
# For gated models (Gemma, Llama), set HF_TOKEN or place it in ~/.hf_token

set -e

MODEL_PRESET=${MODEL_PRESET:-llama3.1-405b}
MODELS_JSON="$(dirname "$0")/../models.json"

if [ ! -f "$MODELS_JSON" ]; then
    echo "ERROR: models.json not found at $MODELS_JSON"
    exit 1
fi

HF_ID=$(python3 -c "import json; print(json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['hf_id'])")
MODEL_DIR="/scratch/project_465002609/models/${MODEL_PRESET}"

echo "Downloading ${HF_ID} to ${MODEL_DIR}..."

pip install --user huggingface-hub 2>/dev/null

# Use HF token for gated models (Gemma, Llama)
if [ -z "$HF_TOKEN" ] && [ -f ~/.hf_token ]; then
    export HF_TOKEN=$(cat ~/.hf_token)
fi

HF_ARGS=""
if [ -n "$HF_TOKEN" ]; then
    HF_ARGS="--token $HF_TOKEN"
    echo "Using HF token for authentication"
fi

huggingface-cli download "$HF_ID" --local-dir "$MODEL_DIR" $HF_ARGS

echo "Done. Model at ${MODEL_DIR}"
ls -lh "$MODEL_DIR"
