#!/bin/bash
# Replay every iter_*.py from the adversarial agent run across the
# 48-cell matrix (N <= 80; N=200/300 cells are skipped to keep this in
# a single wall-time window).
#
# Run this AFTER lumi/run_agent_adversarial.sbatch completes.
#
# Usage:
#   sbatch lumi/replay_candidates_matrix.sbatch
#
#SBATCH --account=project_465002609
#SBATCH --job-name=funwake-replay
#SBATCH --partition=small
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=lumi/logs/replay_matrix_%j.out
#SBATCH --error=lumi/logs/replay_matrix_%j.err

set -e
cd /scratch/project_465002609/julian/funwake_adv
mkdir -p lumi/logs

SCRIPTS_DIR="results_agent_schedule_adversarial"
OUTPUT="results/matrix/adversarial_replay.json"

echo "=== FunWake matrix replay ==="
echo "Start: $(date)"
echo "Scripts: $SCRIPTS_DIR ($(ls $SCRIPTS_DIR/iter_*.py 2>/dev/null | wc -l) candidates)"
echo "Output: $OUTPUT"

pixi run python tools/replay_dir_matrix.py \
    --scripts-dir "$SCRIPTS_DIR" \
    --output "$OUTPUT" \
    --timeout 180 \
    --max-n 80

echo "Done: $(date)"
