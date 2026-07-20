#!/bin/bash
#BSUB -J hnms2[1-16]%10
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/hnms2.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/hnms2.%J.%I.err
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"; export JAX_COMPILATION_CACHE_DIR=/tmp/jax_${LSB_JOBID}_${LSB_JOBINDEX}; export TMPDIR=/tmp
cd $HOME/projects/funwake_unisweep
PYTHONPATH=dependencies/pixwake/src:playground pixi run python tools/run_matrix_ms_highn.py --cell-idx $((LSB_JOBINDEX-1)) --K 50 --out $HOME/clusters_results/funwake_matrix_ms_highn_v2/cell$((LSB_JOBINDEX-1)).json
