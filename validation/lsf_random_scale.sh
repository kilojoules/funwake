#!/bin/bash
#BSUB -J rscale[1-9]%9
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=24GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/rscale.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/rscale.%J.%I.err
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"; export JAX_COMPILATION_CACHE_DIR=/tmp/jax_rs_${LSB_JOBID}_${LSB_JOBINDEX}; export TMPDIR=/tmp
cd $HOME/projects/funwake_unisweep
OUT=$HOME/clusters_results/funwake_random_scale
case $LSB_JOBINDEX in
  1) R=run_matrix_ms;      C=29; TAG=rowp_n80_rosedei ;;
  2) R=run_matrix_ms;      C=35; TAG=rowp_n80_roserowp ;;
  3) R=run_matrix_ms;      C=41; TAG=rowp_n80_roseomnidir ;;
  4) R=run_matrix_ms_highn; C=8;  TAG=rowp_n200_rosedei ;;
  5) R=run_matrix_ms_highn; C=9;  TAG=rowp_n300_rosedei ;;
  6) R=run_matrix_ms_highn; C=10; TAG=rowp_n200_roserowp ;;
  7) R=run_matrix_ms_highn; C=11; TAG=rowp_n300_roserowp ;;
  8) R=run_matrix_ms_highn; C=12; TAG=rowp_n200_roseomnidir ;;
  9) R=run_matrix_ms_highn; C=13; TAG=rowp_n300_roseomnidir ;;
esac
PYTHONPATH=playground/pixwake/src:playground pixi run python tools/$R.py \
    --cell-idx $C --K 50 --only random --out $OUT/${TAG}.random.json
