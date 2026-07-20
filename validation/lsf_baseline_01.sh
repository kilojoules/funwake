#!/bin/bash
#BSUB -J bl01[1-2]%2
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=12GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/bl01.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/bl01.%J.%I.err
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"; export JAX_COMPILATION_CACHE_DIR=/tmp/jax_${LSB_JOBID}_${LSB_JOBINDEX}; export TMPDIR=/tmp
cd $HOME/projects/funwake_unisweep
case $LSB_JOBINDEX in
  1) PROB=results/problem_rowp.json; KEY=rowp_n74 ;;
  2) PROB=playground/problem.json;   KEY=dei_n50 ;;
esac
PYTHONPATH=dependencies/pixwake/src pixi run python tools/run_baseline_batch.py \
    --problem $PROB --seeds 0-499 --out-dir $HOME/clusters_results/funwake_baseline_01tol --out-key $KEY
