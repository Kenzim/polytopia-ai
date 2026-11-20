#!/bin/bash
# A/B of --credit-per-turn: 300-update arms from the champion with the live loop's settings.
# Usage: run.sh <arm name> [extra args...]
set -e
ARM=$1; shift
D=/kvm/tmp/abc
DATA=/kvm/polytopia-ai-data/runs
mkdir -p $D/runs $D/logs
cd /kvm/polytopia/ai/python
export TMPDIR=/kvm/tmp/tmp PYTHONPATH=${ENGINE_DIR:-/kvm/tmp/ptc-engine}:/kvm/polytopia/ai/python OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec nice -n 15 /kvm/polytopia/.venv/bin/python -m polyai.rl --name $ARM --init $DATA/auto_r0077/ckpt_final.pt \
  --ref $DATA/bc3_b6c96_v11/ckpt_final.pt --runs-dir $D/runs --updates 300 --games 384 --rollout-steps 32768 \
  --n-players 2:85,3:10,4:5 --turn-limit 150 --mode might \
  --pool-weights retail:easy=0.025,retail:normal=0.075,retail:hard=0.075,retail:crazy=0.075,bc=0.10,bpolait=0.20,self=0.45 \
  --lr 3e-05 --ent-coef 0.0005 --kl-coef 0.02 --kl-coef-end 0.02 --critic-warmup 2 --snapshot-every 10 \
  --max-snapshots 6 --ckpt-every 20 --adv-filter 0.5 --seed 900 --threads 16 --torch-threads 4 --need-gb 3 \
  --device cuda --tf32 --collector rust --extra-opponents bpolait=$DATA/bpbc_g0412/ckpt_final.pt "$@" \
  >> $D/logs/$ARM.log 2>&1
