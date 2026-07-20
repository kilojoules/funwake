#!/bin/bash
#BSUB -J rs80[1-3]%3
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=24GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/rs80.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/rs80.%J.%I.err
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"; export JAX_COMPILATION_CACHE_DIR=/tmp/jax_rs80_${LSB_JOBID}_${LSB_JOBINDEX}; export TMPDIR=/tmp
cd $HOME/projects/funwake_unisweep
case $LSB_JOBINDEX in
  1) C=29; TAG=rowp_n80_rosedei ;;
  2) C=35; TAG=rowp_n80_roserowp ;;
  3) C=41; TAG=rowp_n80_roseomnidir ;;
esac
PYTHONPATH=dependencies/pixwake/src:playground pixi run python tools/run_matrix_ms.py \
    --cell-idx $C --K 50 --only random --out $HOME/clusters_results/funwake_random_scale/${TAG}.random.json
