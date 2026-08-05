#!/bin/bash
#BSUB -q hpc
#BSUB -J funwake_wk
#BSUB -n 16
#BSUB -R "span[hosts=1] rusage[mem=4GB]"
#BSUB -W 12:00
#BSUB -o /zhome/5c/f/180689/funwake/gbar_queue/worker.%J.out
#BSUB -e /zhome/5c/f/180689/funwake/gbar_queue/worker.%J.err
# Persistent FunWake eval worker: holds one hpc compute node and serves portfolio
# evaluations via a shared-filesystem queue (zhome is shared login<->compute). The
# local orchestrator drops req_<id>.py into gbar_queue/ and reads resp_<id>.json.
# Bootstraps the gbar-native baselines (same-platform) on first start.
set -u
cd ~/funwake
export PIXI_CACHE_DIR=~/.cache/rattler
QDIR=~/funwake/gbar_queue
mkdir -p "$QDIR"
CELLS="dei_n50 dei_n80_omnidir dei_n50_uniform parque_n20 parque_n10_omnidir"
SEEDS="0 1 2"
JOBS=15

# 1. same-platform native baselines (for score_c) if not present
if [ ! -f ~/funwake/gbar_native_baselines.json ]; then
  echo "computing gbar-native baselines..."
  pixi run python funwake2/gbar_eval.py --schedule funwake2/seeds/native.py \
      --cells $CELLS --seeds $SEEDS --steps 8000 \
      --out ~/funwake/gbar_native_baselines.json --jobs $JOBS
fi

# 2. serve: eval each req_<id>.py -> resp_<id>.json (atomic), until wall-time nearly up
echo "READY host=$(hostname) $(date)" > "$QDIR/worker.status"
END=$(( $(date +%s) + 12*3600 - 300 ))
while [ "$(date +%s)" -lt "$END" ]; do
  req=$(ls "$QDIR"/req_*.py 2>/dev/null | head -1)
  if [ -z "$req" ]; then sleep 2; continue; fi
  id=$(basename "$req" .py); id=${id#req_}
  mv "$req" "$QDIR/proc_${id}.py" 2>/dev/null || continue
  pixi run python funwake2/gbar_eval.py --schedule "$QDIR/proc_${id}.py" \
      --cells $CELLS --seeds $SEEDS --steps 8000 \
      --out "$QDIR/resp_${id}.json" --jobs $JOBS >/dev/null 2>&1
  rm -f "$QDIR/proc_${id}.py"
  echo "served $id $(date +%H:%M:%S)" >> "$QDIR/worker.status"
done
echo "EXIT $(date)" >> "$QDIR/worker.status"
