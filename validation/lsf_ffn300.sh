#!/bin/bash
#BSUB -J ffn300
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=24GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 4:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/ffn300.%J.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/ffn300.%J.err
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"; export JAX_COMPILATION_CACHE_DIR=/tmp/jax_ffn300_${LSB_JOBID}; export TMPDIR=/tmp
cd $HOME/projects/funwake_unisweep
PYTHONPATH=dependencies/pixwake/src:playground pixi run python precompute_ff_n300.py
