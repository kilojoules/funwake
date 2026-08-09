#!/bin/bash
#BSUB -q hpc
#BSUB -J fw_heldoutA
#BSUB -n 16
#BSUB -R "span[hosts=1] rusage[mem=4GB]"
#BSUB -W 2:00
#BSUB -o /zhome/5c/f/180689/funwake/gbar_queue/heldoutA.%J.out
#BSUB -e /zhome/5c/f/180689/funwake/gbar_queue/heldoutA.%J.err
# Stage A: evaluate the portfolio generalist (clean it190) + native c*D on a set of
# HELD-OUT farms it190 never trained on, spanning N and wind-rose. Fresh-process,
# parallel. Deltas are computed locally from these two files.
set -u
cd ~/funwake
export PIXI_CACHE_DIR=~/.cache/rattler
CELLS="dei_n100 dei_n120_rosedei parque_n30_uniform rowp_n74 rowp_n74_uniform"
SEEDS="0 1 2 3 4"
echo "=== it190 ==="
pixi run python funwake2/gbar_eval.py --schedule funwake2/state/seed_port_gbar_iter190_clean.py \
    --cells $CELLS --seeds $SEEDS --steps 8000 --out ~/funwake/heldout_it190.json --jobs 16
echo "=== native ==="
pixi run python funwake2/gbar_eval.py --schedule funwake2/seeds/native.py \
    --cells $CELLS --seeds $SEEDS --steps 8000 --out ~/funwake/heldout_native.json --jobs 16
echo "STAGE_A_DONE"
