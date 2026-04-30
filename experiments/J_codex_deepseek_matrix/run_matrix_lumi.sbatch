#!/bin/bash
# LUMI sharded eval: codex+deepseek 12 best scripts x 64 cells x 3 init seeds
# = 2304 total evals. ~4900 cpu-sec/shard at 100 shards: ~80 min walltime per
# shard.
#
# Submit:
#   N_SEEDS=3 NUM_SHARDS=100 sbatch --array=0-99 \
#       experiments/J_codex_deepseek_matrix/run_matrix_lumi.sbatch
#
# Merge with:
#   pixi run python experiments/J_codex_deepseek_matrix/merge_lumi_shards.py

#SBATCH --job-name=funwake_J_matrix
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=lumi/logs/J_matrix_%A_%a.log
#SBATCH --error=lumi/logs/J_matrix_%A_%a.err

set -euo pipefail
cd /scratch/project_465002609/julian/funwake

N_SEEDS=${N_SEEDS:-3}
NUM_SHARDS=${NUM_SHARDS:-100}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

OUT="experiments/J_codex_deepseek_matrix/matrix_lumi_shard${SHARD}.json"

pixi run --frozen python experiments/J_codex_deepseek_matrix/matrix_eval.py \
    --n-seeds "$N_SEEDS" \
    --shard "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --output "$OUT"
