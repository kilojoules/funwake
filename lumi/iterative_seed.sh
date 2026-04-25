#!/bin/bash
# Iterative seed-evolution driver.
#
# For each generation:
#   1. Submit K sbatch jobs (parallel), each running the agent for
#      TIME_BUDGET seconds with the current seed script as hot-start.
#   2. Wait for all K jobs to complete.
#   3. Pick the best script across all K output dirs (highest feasible
#      held-out ROWP AEP; falls back to feasible train if no feasible
#      ROWP; falls back to any train if nothing feasible).
#   4. Update the current-seed symlink to point at the best script.
#   5. Advance to the next generation.
#
# Results are written to results_agent_<model>_gen<G>_s<K>/.
#
# Usage:
#   bash lumi/iterative_seed.sh <model-preset> <num-seeds> <budget-s> <num-gens> [schedule-only]
#
# Examples:
#   bash lumi/iterative_seed.sh deepseek-r1-distill-qwen-7b 4 3600 3
#   SCHEDULE_ONLY=1 bash lumi/iterative_seed.sh qwen2.5-coder-32b 4 3600 3

set -e

MODEL_PRESET=${1:?Usage: $0 <model-preset> <num-seeds> <budget-s> <num-gens>}
NUM_SEEDS=${2:-4}
TIME_BUDGET=${3:-3600}
NUM_GENS=${4:-3}
SCHEDULE_ONLY=${SCHEDULE_ONLY:-}

cd "$(dirname "$0")/.."

MODELS_JSON="$(pwd)/models.json"
NUM_GPUS=$(python3 -c "import json; print(json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['num_gpus'])")
PARTITION=$(python3 -c "import json; print(json.load(open('$MODELS_JSON'))['$MODEL_PRESET']['lumi_partition'])")

MODEL_SLUG="${MODEL_PRESET//./_}"
SEED_DIR="results/iterative_seeds"
mkdir -p "$SEED_DIR"

# Bootstrap seed: if no existing symlink, use the default
if [ -n "$SCHEDULE_ONLY" ]; then
    CURRENT_SEED_LINK="$SEED_DIR/${MODEL_SLUG}_sched_current.py"
else
    CURRENT_SEED_LINK="$SEED_DIR/${MODEL_SLUG}_current.py"
fi
if [ ! -L "$CURRENT_SEED_LINK" ]; then
    if [ -n "$SCHEDULE_ONLY" ]; then
        ln -sf "$(pwd)/results/seed_schedule.py" "$CURRENT_SEED_LINK"
    else
        ln -sf "$(pwd)/results/seed_optimizer.py" "$CURRENT_SEED_LINK"
    fi
fi

echo "=== Iterative Seed Evolution ==="
echo "Model:      $MODEL_PRESET"
echo "Seeds/gen:  $NUM_SEEDS"
echo "Budget:     $((TIME_BUDGET / 60)) min per seed"
echo "Gens:       $NUM_GENS"
echo "Mode:       ${SCHEDULE_ONLY:+schedule-only}${SCHEDULE_ONLY:-full optimizer}"
echo "Seed link:  $CURRENT_SEED_LINK -> $(readlink "$CURRENT_SEED_LINK")"
echo ""

for GEN in $(seq 1 $NUM_GENS); do
    echo ""
    echo "=== Generation $GEN / $NUM_GENS ==="
    echo "  Hot-start: $(readlink "$CURRENT_SEED_LINK")"

    JOBIDS=()
    for SEED in $(seq 1 $NUM_SEEDS); do
        TAG="gen${GEN}"
        JOBID=$(MODEL_PRESET=$MODEL_PRESET TIME_BUDGET=$TIME_BUDGET SEED=$SEED \
            SCHEDULE_ONLY=$SCHEDULE_ONLY EXP_TAG=$TAG \
            HOT_START="$CURRENT_SEED_LINK" \
            sbatch \
                --partition="$PARTITION" \
                --gpus-per-node="$NUM_GPUS" \
                --parsable \
                --export=ALL,MODEL_PRESET,TIME_BUDGET,SEED,SCHEDULE_ONLY,EXP_TAG,HOT_START \
                lumi/run_benchmark.sbatch)
        JOBIDS+=("$JOBID")
        if [ -n "$SCHEDULE_ONLY" ]; then
            OUT_DIR="results_agent_${MODEL_SLUG}_${TAG}_sched_s${SEED}"
        else
            OUT_DIR="results_agent_${MODEL_SLUG}_${TAG}_s${SEED}"
        fi
        echo "  seed $SEED: job $JOBID (output: $OUT_DIR/)"
    done

    # Wait for all jobs in this generation
    JOBS_CSV=$(IFS=,; echo "${JOBIDS[*]}")
    echo "  Waiting for jobs: $JOBS_CSV"
    while true; do
        REMAINING=$(squeue -u "$(whoami)" -j "$JOBS_CSV" -h 2>/dev/null | wc -l | tr -d ' ')
        if [ "$REMAINING" = "0" ]; then
            break
        fi
        sleep 60
    done
    echo "  All seeds for generation $GEN complete."

    # Pick the best script and update the symlink
    if [ -n "$SCHEDULE_ONLY" ]; then
        GLOB="results_agent_${MODEL_SLUG}_gen${GEN}_sched_s*"
    else
        GLOB="results_agent_${MODEL_SLUG}_gen${GEN}_s*"
    fi
    BEST=$(pixi run python tools/pick_best_seed_script.py \
               --dirs $GLOB 2>/dev/null || true)
    if [ -z "$BEST" ]; then
        echo "  WARNING: no best script found for generation $GEN. Keeping previous seed."
    else
        ln -sf "$(pwd)/$BEST" "$CURRENT_SEED_LINK"
        echo "  Generation $GEN best: $BEST"
        echo "  Seed link now: $(readlink "$CURRENT_SEED_LINK")"
    fi
done

echo ""
echo "=== All $NUM_GENS generations complete ==="
echo "Final seed: $(readlink "$CURRENT_SEED_LINK")"
