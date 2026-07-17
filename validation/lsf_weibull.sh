#!/bin/bash
#BSUB -J wbl[1-36]%16
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=14GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/wbl.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/wbl.%J.%I.err
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"; export JAX_COMPILATION_CACHE_DIR=/tmp/jax_${LSB_JOBID}_${LSB_JOBINDEX}; export TMPDIR=/tmp
cd $HOME/projects/funwake_unisweep
IDX=$((LSB_JOBINDEX-1)); CELL=$((IDX/3)); SI=$((IDX%3))
SCHED=(baseline claude gemini); S=${SCHED[$SI]}
PYTHONPATH=playground/pixwake/src:playground pixi run python tools/run_weibull_ms.py \
    --cell-idx $CELL --K 50 --only $S \
    --out $HOME/clusters_results/funwake_weibull/cell${CELL}.${S}.json
