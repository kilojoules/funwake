#!/bin/bash
# LUMI parallel sharded matrix multi-seed eval. Submit as an array job.
#
# Usage:
#   N_SEEDS=3 NUM_SHARDS=20 sbatch --array=0-19 \
#       experiments/D_matrix_multi_seed/run_matrix_multi_seed.sbatch
#
# Each shard writes to a shard-specific JSON; aggregate.py merges.

#SBATCH --job-name=funwake_matrix_seeds
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=lumi/logs/matrix_seeds_%A_%a.log
#SBATCH --error=lumi/logs/matrix_seeds_%A_%a.err

set -euo pipefail
cd /scratch/project_465002609/julian/funwake

N_SEEDS=${N_SEEDS:-3}
NUM_SHARDS=${NUM_SHARDS:-20}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

OUT="experiments/D_matrix_multi_seed/results_shard${SHARD}.json"

pixi run python experiments/D_matrix_multi_seed/eval.py \
    --n-seeds "$N_SEEDS" \
    --shard "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --output "$OUT"
