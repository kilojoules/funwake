#!/bin/bash
# LUMI parallel sharded matrix eval for the 3 codex sched-run best schedules.
#
# Submit as an array job:
#   N_SEEDS=3 NUM_SHARDS=20 sbatch --array=0-19 \
#       experiments/H_codex_six_runs/run_matrix_codex_lumi.sbatch
#
# Each shard writes to a shard-specific JSON; merge with:
#   pixi run python tools/merge_random_search_shards.py \
#       --pattern 'experiments/H_codex_six_runs/matrix_lumi_shard*.json' \
#       --output experiments/H_codex_six_runs/matrix_lumi_results.json
# (or wire in a dedicated merge — see Notes below.)

#SBATCH --job-name=funwake_codex_matrix
#SBATCH --account=project_465002609
#SBATCH --partition=small
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=lumi/logs/codex_matrix_%A_%a.log
#SBATCH --error=lumi/logs/codex_matrix_%A_%a.err

set -euo pipefail
cd /scratch/project_465002609/julian/funwake

N_SEEDS=${N_SEEDS:-3}
NUM_SHARDS=${NUM_SHARDS:-20}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

OUT="experiments/H_codex_six_runs/matrix_lumi_shard${SHARD}.json"

pixi run python experiments/H_codex_six_runs/matrix_eval.py \
    --n-seeds "$N_SEEDS" \
    --shard "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --output "$OUT"
