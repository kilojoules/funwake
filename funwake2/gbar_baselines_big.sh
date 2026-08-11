#!/bin/bash
#BSUB -q hpc
#BSUB -J fw_baseBig
#BSUB -n 16
#BSUB -R "span[hosts=1] rusage[mem=4GB]"
#BSUB -W 3:00
#BSUB -o /zhome/5c/f/180689/funwake/gbar_queue/baseBig.%J.out
#BSUB -e /zhome/5c/f/180689/funwake/gbar_queue/baseBig.%J.err
# Native c*D baselines (gbar, same-platform) for the large-portfolio scaling
# experiment: 14 TRAIN + 6 HELD-OUT farms spanning N x wind-rose. -> big baselines.
set -u
cd ~/funwake
export PIXI_CACHE_DIR=~/.cache/rattler
CELLS="dei_n30_rosedei dei_n30_roseuniform dei_n40_roseomnidir dei_n50_rosedei dei_n50_roseuniform dei_n60_roserowp dei_n60_roseomnidir dei_n70_rosedei dei_n80_roseomnidir dei_n80_roseuniform dei_n100_rosedei parque_n10_omnidir parque_n20 parque_n30_uniform dei_n40_rosedei dei_n70_roseuniform dei_n120_rosedei dei_n50_roseomnidir parque_n14_uniform rowp_n74"
pixi run python funwake2/gbar_eval.py --schedule funwake2/seeds/native.py \
    --cells $CELLS --seeds 0 1 2 --steps 8000 --out ~/funwake/gbar_baselines_big.json --jobs 16
echo BASELINES_BIG_DONE
