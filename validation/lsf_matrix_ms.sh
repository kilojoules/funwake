#!/bin/bash
#BSUB -J mxms[1-48]%10
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=12GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/mxms.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/mxms.%J.%I.err

# Fair matrix multistart: each cell (0..47) runs baseline/claude/gemini
# schedule_fn × K=50 init seeds through the funwake skeleton, best feasible.
# Makes schedule-vs-baseline apples-to-apples (matched starts, same skeleton).
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache_${LSB_JOBID}_${LSB_JOBINDEX}
export TMPDIR=/tmp

PROJ=$HOME/projects/funwake_unisweep
cd "$PROJ"
CELL=$((LSB_JOBINDEX - 1))
OUT=$HOME/clusters_results/funwake_matrix_ms/cell${CELL}.json

echo "=== matrix MS cell $CELL ==="; hostname; date
PYTHONPATH=dependencies/pixwake/src:playground pixi run python \
    tools/run_matrix_ms.py --cell-idx "$CELL" --K 50 --out "$OUT"
echo "done"; date
