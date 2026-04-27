#!/bin/bash
# 500-multistart matrix eval for the 3 codex sched-run best schedules.
# Apples-to-apples vs the 500-multistart baseline used in Fig 3.
#
# Sharding: array job. Each shard handles a hash-distributed subset of
# the (label, cell, seed) keyspace.
#
# Submit:
#   N_SEEDS=500 NUM_SHARDS=100 sbatch --array=0-99 \
#       experiments/H_codex_six_runs/run_matrix_codex_500start_lumi.sbatch
#
# Merge after completion:
#   pixi run python experiments/H_codex_six_runs/merge_lumi_500_shards.py

#SBATCH --job-name=funwake_codex_matrix_500
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=lumi/logs/codex_matrix_500_%A_%a.log
#SBATCH --error=lumi/logs/codex_matrix_500_%A_%a.err

set -euo pipefail
cd /scratch/project_465002609/julian/funwake

N_SEEDS=${N_SEEDS:-500}
NUM_SHARDS=${NUM_SHARDS:-100}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

OUT="experiments/H_codex_six_runs/matrix_lumi_500_shard${SHARD}.json"

pixi run python experiments/H_codex_six_runs/matrix_eval.py \
    --n-seeds "$N_SEEDS" \
    --shard "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --output "$OUT"
