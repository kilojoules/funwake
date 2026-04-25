#!/bin/bash
# Run 500-start baselines and schedule evaluations for N=200, N=300
# on both DEI and ROWP polygons across all 4 wind roses.
#
# Usage:
#   sbatch lumi/run_matrix_n200_n300.sbatch
#
# This is CPU-only (no GPU needed). Runs baselines first, then
# evaluates top LLM schedules on the new cells.

#SBATCH --job-name=funwake-n200-n300
#SBATCH --account=project_465002609
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --output=lumi/logs/matrix_n200_n300_%j.log
#SBATCH --error=lumi/logs/matrix_n200_n300_%j.err

set -e
cd /scratch/project_465002609/julian/funwake
mkdir -p lumi/logs

export JAX_ENABLE_X64=True

# ── Schedules to evaluate ──
CLAUDE_SCHED="results_agent_schedule_only_5hr/iter_192.py"
GEMINI_SCHED="results_agent_gemini_cli_5hr/iter_118.py"

# ── Parameters ──
FARMS="dei rowp"
NS="200 300"
ROSES="uniform omnidir dei rowp"
BASELINE_STARTS=500
SCHED_TIMEOUT=1200   # 20 min per schedule eval at large N

echo "=== FunWake N=200/300 Matrix ==="
echo "Start: $(date)"

# ── Phase 1: 500-start baselines ──
echo ""
echo "=== PHASE 1: Baselines ==="
for farm in $FARMS; do
  for n in $NS; do
    for rose in $ROSES; do
      problem="results/matrix/problem_${farm}_n${n}_rose${rose}.json"
      if [ ! -f "$problem" ]; then
        echo "SKIP (missing): $problem"
        continue
      fi

      outfile="results/matrix/baseline_${farm}_n${n}_rose${rose}.json"
      if [ -f "$outfile" ]; then
        echo "SKIP (exists): $outfile"
        continue
      fi
      echo "Baseline: ${farm}_n${n}_rose${rose} (${BASELINE_STARTS} starts)"
      pixi run python tools/compute_baselines.py \
        --problems "$problem" \
        --n-starts $BASELINE_STARTS \
        --output "$outfile" \
        2>&1 | tail -3
    done
  done
done

# ── Phase 2: Schedule evaluations ──
echo ""
echo "=== PHASE 2: Schedule evaluations ==="
for sched_name in claude gemini; do
  if [ "$sched_name" = "claude" ]; then
    sched="$CLAUDE_SCHED"
  else
    sched="$GEMINI_SCHED"
  fi

  if [ ! -f "$sched" ]; then
    echo "SKIP (missing schedule): $sched"
    continue
  fi

  for farm in $FARMS; do
    for n in $NS; do
      for rose in $ROSES; do
        problem="results/matrix/problem_${farm}_n${n}_rose${rose}.json"
        if [ ! -f "$problem" ]; then
          continue
        fi

        echo "${sched_name}: ${farm}_n${n}_rose${rose}"
        pixi run python tools/run_optimizer.py "$sched" \
          --problem "$problem" \
          --timeout $SCHED_TIMEOUT \
          --schedule-only \
          --log "results/matrix/schedules_n200_n300.json" \
          2>&1 | tail -1
      done
    done
  done
done

echo ""
echo "=== DONE ==="
echo "End: $(date)"
