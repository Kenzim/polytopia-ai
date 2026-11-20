#!/bin/bash
# The --overlap A/B: when the site A/B arms have finished, trains abf_ovl (--features --overlap, otherwise as the
# control abd_feat) beside the temperature arms; when it and the temperature matches are done, plays its matches.
D=/kvm/polytopia-ai-data/ab; L=/kvm/polytopia-ai-data/runs; P=/kvm/polytopia-ai-data/ptai
until [ -f $P/runs/abe_ctrl/ckpt_final.pt ] && [ -f $P/runs/abe_site/ckpt_final.pt ]; do sleep 30; done
echo "$(date -u +%FT%TZ) abf_ovl start"
/kvm/ab/run.sh abf_ovl --features --overlap
echo "$(date -u +%FT%TZ) abf_ovl exit $?"
until grep -q "ALL DONE" $D/eval-abf/done.txt 2>/dev/null; do sleep 30; done
ENGINE_DIR=/kvm/ab/ptc-engine2 AI_DIR=/kvm/ab/ai DEVICE=cuda JOBS=3 /kvm/ab/ai/tools/ab_eval.sh $D/eval-ovl \
  r77=$L/auto_r0077/ckpt_final.pt feat=$D/runs/abd_feat/ckpt_final.pt ovl=$D/runs/abf_ovl/ckpt_final.pt
