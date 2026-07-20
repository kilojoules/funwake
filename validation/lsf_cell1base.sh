#!/bin/bash
#BSUB -J c1base
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 48:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/c1base.%J.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/c1base.%J.err
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"; export JAX_COMPILATION_CACHE_DIR=/tmp/jax_c1base_${LSB_JOBID}; export TMPDIR=/tmp
cd $HOME/projects/funwake_unisweep
PYTHONPATH=dependencies/pixwake/src:playground pixi run python tools/run_matrix_ms_highn.py \
    --cell-idx 1 --K 50 --only baseline \
    --out $HOME/clusters_results/funwake_highn_k50/cell1.baseline.json
