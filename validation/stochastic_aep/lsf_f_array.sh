#!/bin/bash
#BSUB -J fcost[1-30]%10
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=12GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/fcost.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/fcost.%J.%I.err

# Task F — baseline ES-cost per cell. 30 multidir cells × {ES-on, ES-off} × 3 seeds = 6 runs/cell.
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache_${LSB_JOBID}_${LSB_JOBINDEX}
export TMPDIR=/tmp

PROJ=$HOME/projects/funwake_unisweep
cd "$PROJ"

CELL_IDX=$((LSB_JOBINDEX - 1))
OUT_DIR="$HOME/clusters_results/funwake_taskf"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/cell${CELL_IDX}.json"

echo "=== Task F task $LSB_JOBINDEX, cell_idx=$CELL_IDX ==="
hostname; date

PYTHONPATH=dependencies/pixwake/src:validation/stochastic_aep pixi run python \
    validation/stochastic_aep/run_f_es_cost_per_cell.py \
    --cell-idx "$CELL_IDX" --out "$OUT"

echo "Done."; date
