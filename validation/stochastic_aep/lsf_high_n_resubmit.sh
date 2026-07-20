#!/bin/bash
#BSUB -J highn_re[5,6,7,13,15]
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 48:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/highn_re.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/highn_re.%J.%I.err

# Resubmit partial high-N cells with 48h walltime.
# LSF task indices 5, 6, 7, 13, 15 → cell_idx 4, 5, 6, 12, 14.
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache_${LSB_JOBID}_${LSB_JOBINDEX}
export TMPDIR=/tmp

PROJ=$HOME/projects/funwake_unisweep
cd "$PROJ"

CELL_IDX=$((LSB_JOBINDEX - 1))
OUT_DIR="$HOME/clusters_results/funwake_highn"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/cell${CELL_IDX}.json"

echo "=== highn RE task $LSB_JOBINDEX / 16, cell_idx=$CELL_IDX (resume) ==="
hostname; date

PYTHONPATH=dependencies/pixwake/src:validation/stochastic_aep pixi run python \
    validation/stochastic_aep/run_high_n_per_cell.py \
    --cell-idx "$CELL_IDX" --out "$OUT"

echo "Done."; date
