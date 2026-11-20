#!/bin/bash
# Head-to-head gaps between the four Play our AI levels (ledger to-do 8e10c9d2): adjacent levels only, 400 games each,
# top move, 1v1 Might, on fresh seeds. Starts when the value-opp A/B matches (ab/eval-abg) are done.
O=/kvm/polytopia-ai-data/ab/eval-levels; R=/kvm/polytopia-ai-data/runs
until grep -q "ALL DONE" /kvm/polytopia-ai-data/ab/eval-abg/done.txt 2>/dev/null; do sleep 60; done
export PYTHONPATH=/kvm/ab2/ext:/kvm/ab2/ai/python OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMPDIR=/kvm/tmp/tmp
cd /kvm/ab2/ai/python
m() { nice -n 15 /kvm/polytopia/.venv/bin/python -m polyai.h2h --a $2 --b $3 --seeds 200 --seed-start 3200000 \
  --mode might --threads 8 --device cuda --out $O/$1.jsonl >> $O/$1.log 2>&1; echo "$(date -u +%FT%TZ) $1 exit $?" >> $O/done.txt; }
m r77_v_r60 $R/auto_r0077/ckpt_final.pt $R/auto_r0060/ckpt_final.pt &
m r60_v_rl3 $R/auto_r0060/ckpt_final.pt $R/rl3_b6c96/ckpt_u00480.pt &
m rl3_v_bc3 $R/rl3_b6c96/ckpt_u00480.pt $R/bc3_b6c96_v11/ckpt_final.pt &
wait; echo "$(date -u +%FT%TZ) ALL DONE" >> $O/done.txt
