#!/bin/bash
#BSUB -J hnbl[1-16]%8
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u /dev/null
#BSUB -o /zhome/5c/f/180689/clusters_logs/hnbl.%J.%I.out
#BSUB -e /zhome/5c/f/180689/clusters_logs/hnbl.%J.%I.err

# High-N TopFarm baseline: 500-multistart SGD on each N=200/300 cell (0..15),
# matching the low-N baselines_matrix (best-of-500). Cell -> problem via a
# fixed enumeration identical to run_matrix_ms_highn's CELLS order.
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache_${LSB_JOBID}_${LSB_JOBINDEX}
export TMPDIR=/tmp
PROJ=$HOME/projects/funwake_unisweep; cd "$PROJ"

CELL=$((LSB_JOBINDEX - 1))
# CELLS = [(f,n,r) for f in [dei,rowp] for r in [dei,rowp,omnidir,uniform] for n in [200,300]]
FARMS=(dei rowp); ROSES=(dei rowp omnidir uniform); NS=(200 300)
fi=$(( CELL / 8 )); rem=$(( CELL % 8 )); ri=$(( rem / 2 )); ni=$(( rem % 2 ))
KEY="${FARMS[$fi]}_n${NS[$ni]}_rose${ROSES[$ri]}"
PROB="results/matrix/problem_${KEY}.json"
OUT=$HOME/clusters_results/funwake_baseline_highn

echo "=== high-N baseline cell $CELL -> $KEY ==="; hostname; date
PYTHONPATH=playground/pixwake/src pixi run python \
    tools/run_baseline_batch.py \
    --problem "$PROB" --seeds 0-49 --out-dir "$OUT" --out-key "$KEY"
echo done; date
