#!/bin/bash
# One A/B arm on poly4090 beside the live loop: 300 updates from round 77 with the live loop's settings, on the
# A/B2 code (/kvm/ab2/ai, ai 47167b5) and profile-guided bindings (/kvm/ab2/ext, engine 80a68d0); the live loop's checkout and venv packages
# are not used. Usage: run.sh <arm name> [extra polyai.rl arguments...]
set -e
ARM=$1; shift
D=/kvm/polytopia-ai-data/ab
DATA=/kvm/polytopia-ai-data/runs
cd /kvm/ab2/ai/python
export PYTHONPATH=/kvm/ab2/ext:/kvm/ab2/ai/python OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec nice -n 15 /kvm/polytopia/.venv/bin/python -m polyai.rl --name $ARM --init $DATA/auto_r0077/ckpt_final.pt \
  --ref $DATA/bc3_b6c96_v11/ckpt_final.pt --runs-dir $D/runs --updates 300 --games 384 --rollout-steps 32768 \
  --n-players 2:85,3:10,4:5 --turn-limit 150 --mode might \
  --pool-weights retail:easy=0.025,retail:normal=0.075,retail:hard=0.075,retail:crazy=0.075,bc=0.10,bpolait=0.20,self=0.45 \
  --lr 3e-05 --ent-coef 0.0005 --kl-coef 0.02 --kl-coef-end 0.02 --critic-warmup 2 --snapshot-every 10 \
  --max-snapshots 6 --ckpt-every 20 --adv-filter 0.5 --seed 900 --threads 16 --torch-threads 4 --need-gb 3 \
  --device cuda --tf32 --collector rust --extra-opponents bpolait=$DATA/bpbc_g0412/ckpt_final.pt "$@" \
  >> $D/logs/$ARM.log 2>&1
