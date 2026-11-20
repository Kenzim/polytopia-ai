#!/bin/bash
# When the two arms of the site A/B have finished: starts the temperature A/B (control abd_feat), plays the site A/B
# matches (each other, round 77, Crazy) and 200 games against the PolytopAI site for each site-A/B arm and round 77.
P=/kvm/polytopia-ai-data/ptai; L=/kvm/polytopia-ai-data/runs; PY=/kvm/polytopia/.venv/bin/python
until [ -f $P/runs/abe_ctrl/ckpt_final.pt ] && [ -f $P/runs/abe_site/ckpt_final.pt ]; do sleep 30; done
date -u +"%F %T site A/B arms done; temperature arms start" >> $P/logs/start.log
nohup /kvm/ab/run.sh abf_t05 --features --temperature 0.5 > /dev/null 2>&1 &
nohup /kvm/ab/run.sh abf_t05o --features --temperature 0.5 --opp-temperature 0.5 > /dev/null 2>&1 &
(
  cd /kvm/ptai/ai/python
  export PYTHONPATH=/kvm/ptai/engine:/kvm/ptai/ai/python OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR=/kvm/tmp/tmp
  mkdir -p $P/eval
  for m in abe_site=$P/runs/abe_site abe_ctrl=$P/runs/abe_ctrl r77=$L/auto_r0077; do
    nice -n 15 $PY -m polyai.polytopai --ckpt ${m#*=}/ckpt_final.pt --games 200 --parallel 8 --level 1 --device cuda \
      --out $P/eval/site_${m%%=*}.jsonl >> $P/eval/site_${m%%=*}.log 2>&1
    echo "$(date -u +%FT%TZ) site_${m%%=*} exit $?" >> $P/eval/done_site.txt
  done
  echo "$(date -u +%FT%TZ) ALL DONE" >> $P/eval/done_site.txt
) &
ENGINE_DIR=/kvm/ptai/engine AI_DIR=/kvm/ptai/ai DEVICE=cuda JOBS=3 /kvm/ptai/ai/tools/ab_eval.sh $P/eval \
  r77=$L/auto_r0077/ckpt_final.pt ctrl=$P/runs/abe_ctrl/ckpt_final.pt site=$P/runs/abe_site/ckpt_final.pt
wait
