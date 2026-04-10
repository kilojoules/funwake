#!/bin/bash
# Download a model from HuggingFace to LUMI scratch.
# Run from a login node (has internet access).
#
# Usage:
#   MODEL_PRESET=qwen2.5-72b bash lumi/download_model.sh
#   MODEL_PRESET=llama3.3-70b bash lumi/download_model.sh

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

huggingface-cli download "$HF_ID" --local-dir "$MODEL_DIR"

echo "Done. Model at ${MODEL_DIR}"
ls -lh "$MODEL_DIR"
