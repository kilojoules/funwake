#!/bin/bash
#BSUB -J hnk50[1-48]%16
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/hnk50.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/hnk50.%J.%I.err
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"; export JAX_COMPILATION_CACHE_DIR=/tmp/jax_${LSB_JOBID}_${LSB_JOBINDEX}; export TMPDIR=/tmp
cd $HOME/projects/funwake_unisweep
IDX=$((LSB_JOBINDEX-1)); CELL=$((IDX/3)); SI=$((IDX%3))
SCHED=(baseline claude gemini); S=${SCHED[$SI]}
OUT=$HOME/clusters_results/funwake_highn_k50/cell${CELL}.${S}.json
mkdir -p $HOME/clusters_results/funwake_highn_k50
PYTHONPATH=dependencies/pixwake/src:playground pixi run python tools/run_matrix_ms_highn.py \
    --cell-idx $CELL --K 50 --only $S --out $OUT
