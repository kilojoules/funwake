#!/bin/bash
# Random-search ablation sharded across 8 GCDs on one standard-g node.
# Each GCD runs 40 samples with a distinct Python RNG seed, so together
# they produce 320 unique samples from the parameterized schedule family
# (matching Claude's 320-attempt schedule-only budget).
#
# Total work: 320 samples * ~75s (train+ROWP) / 8 shards = ~50 min wall.
# Budget a safety buffer for JAX JIT warmup and any stragglers.
#
# Usage:
#   sbatch lumi/ablation_random_search.sbatch
#
# Result merge:
#   pixi run python tools/merge_random_search_shards.py \
#       --shard-dirs results_random_search_shard_{0..7} \
#       --output-dir results_random_search_320

#SBATCH --job-name=rs-abl
#SBATCH --account=project_465002609
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --output=lumi/logs/rs_abl_%j.log
#SBATCH --error=lumi/logs/rs_abl_%j.err

set -e
cd /scratch/project_465002609/julian/funwake
mkdir -p lumi/logs

SAMPLES_PER_SHARD=${SAMPLES_PER_SHARD:-40}
TIMEOUT=${TIMEOUT:-120}

export JAX_ENABLE_X64=True
export PYTHONUNBUFFERED=1

echo "=== Random Search Ablation ==="
echo "Shards: 8, samples per shard: $SAMPLES_PER_SHARD"
echo "Total samples: $((8 * SAMPLES_PER_SHARD))"
echo "Per-sample timeout: ${TIMEOUT}s"
echo ""

for gcd in 0 1 2 3 4 5 6 7; do
    (
        export ROCR_VISIBLE_DEVICES=$gcd
        export HIP_VISIBLE_DEVICES=$gcd
        OUT="results_random_search_shard_${gcd}"
        LOG="lumi/logs/rs_abl_shard_${gcd}.log"
        echo "[$gcd] starting, log=$LOG"
        pixi run python tools/random_search_ablation.py \
            --n-samples $SAMPLES_PER_SHARD \
            --output-dir "$OUT" \
            --timeout "$TIMEOUT" \
            --seed "$gcd" \
            > "$LOG" 2>&1
        echo "[$gcd] done"
    ) &
done
wait

echo ""
echo "All shards complete. Merge with:"
echo "  pixi run python tools/merge_random_search_shards.py \\"
echo "      --shard-dirs results_random_search_shard_{0..7} \\"
echo "      --output-dir results_random_search_320"
