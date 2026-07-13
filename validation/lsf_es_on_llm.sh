#!/bin/bash
#BSUB -J esllm[1-4]%4
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "rusage[mem=24GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/esllm.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/esllm.%J.%I.err
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"; export JAX_COMPILATION_CACHE_DIR=/tmp/jax_esllm_${LSB_JOBID}_${LSB_JOBINDEX}; export TMPDIR=/tmp
cd $HOME/projects/funwake_unisweep
SCHED=(claude gemini); CIDX=(10 11); TAGS=(rowp_n200_roserowp rowp_n300_roserowp)
IDX=$((LSB_JOBINDEX-1)); SI=$((IDX/2)); CI=$((IDX%2))
S=${SCHED[$SI]}; C=${CIDX[$CI]}; TAG=${TAGS[$CI]}
PYTHONPATH=playground/pixwake/src:playground pixi run python tools/run_matrix_ms_highn.py \
    --cell-idx $C --K 50 --only $S --es \
    --out $HOME/clusters_results/funwake_es_on_llm/${TAG}.${S}_es.json
