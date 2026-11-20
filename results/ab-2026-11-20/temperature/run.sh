#!/bin/bash
# Round 77 top move against round 77 sampling, at temperature 1.0 and 0.5: which kind of play is stronger.
O=/kvm/polytopia-ai-data/ab/eval-temp; R=/kvm/polytopia-ai-data/runs/auto_r0077/ckpt_final.pt
export PYTHONPATH=/kvm/ab/ptc-engine2:/kvm/ab/ai/python OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMPDIR=/kvm/tmp/tmp
cd /kvm/ab/ai/python
for t in 1.0 0.5; do
  nice -n 15 /kvm/polytopia/.venv/bin/python -m polyai.h2h --a $R --b $R:sample=$t --seeds 200 --seed-start 3100000 \
    --mode might --threads 8 --device cuda --out $O/greedy_v_t$t.jsonl >> $O/greedy_v_t$t.log 2>&1 &
done
wait
echo "$(date -u +%FT%TZ) ALL DONE" >> $O/done.txt
