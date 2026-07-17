#!/bin/bash
#BSUB -J blredo[1-10]
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=12GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 8:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/blredo.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/blredo.%J.%I.err

# Baseline redo under current env lineage: 500-multistart SGD on the DEI
# training farm (playground/problem_train.json), 50 seeds per array task.
# Purpose: close the Apr-9 pixi-env drift audit — the committed 5540.72
# baseline was computed pre-drift; every recent train gap needs a
# same-env baseline.
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache_${LSB_JOBID}_${LSB_JOBINDEX}
export TMPDIR=/tmp

PROJ=$HOME/projects/funwake_unisweep
cd "$PROJ"

LO=$(( (LSB_JOBINDEX - 1) * 50 ))
HI=$(( LSB_JOBINDEX * 50 - 1 ))
OUT=$HOME/clusters_results/funwake_baseline_redo

echo "=== baseline redo task $LSB_JOBINDEX seeds $LO-$HI ==="
hostname; date

PYTHONPATH=playground/pixwake/src pixi run python \
    tools/run_baseline_batch.py \
    --problem playground/problem_train.json \
    --seeds ${LO}-${HI} \
    --out-dir "$OUT" \
    --out-key train_dei_n50

echo "done"; date
