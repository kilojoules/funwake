#!/bin/bash
#BSUB -J unisweep[1-12]%6
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 6:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/unisweep.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/unisweep.%J.%I.err

# Isolate JAX compilation cache per task (avoid stale AOT entries)
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache_$LSB_JOBID_$LSB_JOBINDEX
export TMPDIR=/tmp

# Uniform-rose ηT sweep — 1 cell per array task.
# 12 tasks × (7 ηT × 3 seeds = 21 runs each).

set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"

PROJ=$HOME/projects/funwake_unisweep
cd "$PROJ"

# Initialize pixi env if not yet (only first task does this; others wait & reuse)
if [ ! -d ".pixi" ]; then
    pixi install
fi

# Cell index 0..11 = LSB_JOBINDEX - 1
CELL_IDX=$((LSB_JOBINDEX - 1))
OUT_DIR="$HOME/clusters_results/funwake_unisweep"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/cell${CELL_IDX}.json"

echo "=== unisweep task $LSB_JOBINDEX / 12, cell_idx=$CELL_IDX ==="
echo "host: $(hostname)"
date

PYTHONPATH=playground/pixwake/src:validation/stochastic_aep pixi run python \
    validation/stochastic_aep/run_uniform_per_cell.py \
    --cell-idx "$CELL_IDX" --out "$OUT"

echo "Done."
date
