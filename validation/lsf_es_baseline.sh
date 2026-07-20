#!/bin/bash
#BSUB -J esbl[1-12]%12
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=24GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/esbl.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/esbl.%J.%I.err
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"; export JAX_COMPILATION_CACHE_DIR=/tmp/jax_esbl_${LSB_JOBID}_${LSB_JOBINDEX}; export TMPDIR=/tmp
cd $HOME/projects/funwake_unisweep
ROSES=(uniform rowp omnidir); NS=(50 80 200 300)
IDX=$((LSB_JOBINDEX-1)); RI=$((IDX/4)); NI=$((IDX%4))
R=${ROSES[$RI]}; N=${NS[$NI]}; KEY=rowp_n${N}_rose${R}
PYTHONPATH=dependencies/pixwake/src pixi run python tools/run_baseline_batch.py \
    --problem results/matrix/problem_${KEY}.json --seeds 0-49 \
    --early-stopping --es-threshold 0.1 \
    --out-dir $HOME/clusters_results/funwake_es_baseline --out-key ${KEY}_es
