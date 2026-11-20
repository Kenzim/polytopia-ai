#!/bin/bash
# When the temperature arms have finished: their 400-game matches against the control abd_feat, each other,
# round 77 and Crazy, on the same code, bindings and device as the features A/B.
D=/kvm/polytopia-ai-data/ab; L=/kvm/polytopia-ai-data/runs
until [ -f $D/runs/abf_t05/ckpt_final.pt ] && [ -f $D/runs/abf_t05o/ckpt_final.pt ]; do sleep 30; done
ENGINE_DIR=/kvm/ab/ptc-engine2 AI_DIR=/kvm/ab/ai DEVICE=cuda JOBS=5 /kvm/ab/ai/tools/ab_eval.sh $D/eval-abf \
  r77=$L/auto_r0077/ckpt_final.pt feat=$D/runs/abd_feat/ckpt_final.pt t05=$D/runs/abf_t05/ckpt_final.pt \
  t05o=$D/runs/abf_t05o/ckpt_final.pt
