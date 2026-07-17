#!/bin/bash
#BSUB -J hnms[1-16]%8
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/hnms.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/hnms.%J.%I.err

# High-N schedules multistart: cell 0..15 (N=200/300), baseline/claude/gemini
# schedule_fn x K=50 starts through the funwake skeleton, best feasible.
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache_${LSB_JOBID}_${LSB_JOBINDEX}
export TMPDIR=/tmp
PROJ=$HOME/projects/funwake_unisweep; cd "$PROJ"
CELL=$((LSB_JOBINDEX - 1))
OUT=$HOME/clusters_results/funwake_matrix_ms_highn/cell${CELL}.json
echo "=== high-N sched MS cell $CELL ==="; hostname; date
PYTHONPATH=playground/pixwake/src:playground pixi run python \
    tools/run_matrix_ms_highn.py --cell-idx "$CELL" --K 50 --out "$OUT"
echo done; date
