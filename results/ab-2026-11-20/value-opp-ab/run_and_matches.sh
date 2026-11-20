#!/bin/bash
# The --value-opp A/B: when the temperature and overlap arms have finished, trains abg_ctrl (--features, as the loop
# from round 99) and abg_vopp (+ --value-opp 2.0 --value-opp-frac 0.25), both on /kvm/ab2 (same build), then plays
# their matches against each other, round 77 and Crazy.
D=/kvm/polytopia-ai-data/ab; L=/kvm/polytopia-ai-data/runs
until [ -f $D/runs/abf_t05/ckpt_final.pt ] && [ -f $D/runs/abf_t05o/ckpt_final.pt ] && [ -f $D/runs/abf_ovl/ckpt_final.pt ]; do sleep 30; done
echo "$(date -u +%FT%TZ) abg arms start"
/kvm/ab2/run.sh abg_ctrl --features &
/kvm/ab2/run.sh abg_vopp --features --value-opp 2.0 --value-opp-frac 0.25 &
wait
echo "$(date -u +%FT%TZ) abg arms done"
ENGINE_DIR=/kvm/ab2/ext AI_DIR=/kvm/ab2/ai DEVICE=cuda JOBS=3 /kvm/ab2/ai/tools/ab_eval.sh $D/eval-abg \
  r77=$L/auto_r0077/ckpt_final.pt ctrl=$D/runs/abg_ctrl/ckpt_final.pt vopp=$D/runs/abg_vopp/ckpt_final.pt
