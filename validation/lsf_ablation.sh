#!/bin/bash
#BSUB -J abl[1-6]%6
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=24GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/abl.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/abl.%J.%I.err
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"; export JAX_COMPILATION_CACHE_DIR=/tmp/jax_abl_${LSB_JOBID}_${LSB_JOBINDEX}; export TMPDIR=/tmp
cd $HOME/projects/funwake_unisweep
VARIANTS=(ab_nobumps ab_weakalpha ab_stdbetas); CELLS_IDX=(10 11); TAGS=(rowp_n200_roserowp rowp_n300_roserowp)
IDX=$((LSB_JOBINDEX-1)); VI=$((IDX/2)); CI=$((IDX%2))
V=${VARIANTS[$VI]}; C=${CELLS_IDX[$CI]}; TAG=${TAGS[$CI]}
PYTHONPATH=dependencies/pixwake/src:playground pixi run python tools/run_matrix_ms_highn.py \
    --cell-idx $C --K 50 --only $V --out $HOME/clusters_results/funwake_ablation/${TAG}.${V}.json
