#!/bin/bash
# Run 500-multi-start baselines for the 48-cell matrix on a LUMI
# CPU node. Uses Python multiprocessing (tools/run_matrix_baselines_parallel.py)
# to run all 48 problems concurrently; each problem's worker uses
# run_baseline_batch.py to amortize JAX JIT compile across its 500 seeds.
#
# Usage:
#   sbatch lumi/matrix_baselines_cpu.sbatch

#SBATCH --job-name=mxbase-cpu
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=lumi/logs/mxbase_cpu_%j.log
#SBATCH --error=lumi/logs/mxbase_cpu_%j.err

set -e
cd /scratch/project_465002609/julian/funwake
mkdir -p lumi/logs/matrix_baselines

export JAX_ENABLE_X64=True
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu

N_SEEDS=${N_SEEDS:-500}
WORKERS=${WORKERS:-64}

pixi run python tools/run_matrix_baselines_parallel.py \
    --manifest results/matrix/manifest.json \
    --out-dir lumi/logs/matrix_baselines \
    --n-seeds "$N_SEEDS" \
    --workers "$WORKERS"

echo ""
echo "Aggregate with:"
echo "  pixi run python tools/aggregate_matrix_baselines.py \\"
echo "      --matrix-dir results/matrix \\"
echo "      --out results/matrix/baselines_matrix.json"
