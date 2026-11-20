#!/bin/bash
# Waits for the two arms, then plays their 400-game matches (each other, the control arm, round 77, retail Crazy).
D=/kvm/polytopia-ai-data/ab; L=/kvm/polytopia-ai-data/runs
until [ -f $D/runs/abd_feat/ckpt_final.pt ] && [ -f $D/runs/abd_noadv/ckpt_final.pt ]; do sleep 30; done
ENGINE_DIR=/kvm/ab/ptc-engine2 AI_DIR=/kvm/ab/ai DEVICE=cuda JOBS=5 /kvm/ab/ai/tools/ab_eval.sh $D/eval \
  r77=$L/auto_r0077/ckpt_final.pt ctrl=$D/ref/abc_ctrl/ckpt_final.pt feat=$D/runs/abd_feat/ckpt_final.pt \
  noadv=$D/runs/abd_noadv/ckpt_final.pt
