#!/bin/bash
# Seventh A/B on /kvm/ab2 (same build as abl), recipe abk_1v1 (loop pool, two-player games only, features, seed 900),
# which the live loop uses from round 104.
# Why: each update sees about 35 finished games (32,768 learner decisions, 1,637 a game), so the win/loss signal per
# update is thin; the learning rate is ruled out (1e-4 collapses, 1e-5 equals 3e-5), and the batch size has never
# been varied (docs/research.md, recommendation 3).
# abm_b64k: abk_1v1 with --rollout-steps 65536 (twice the games per update), 300 updates from round 77.
# abm_u600: abk_1v1 continued for 300 more updates (600 in all at 32,768): the same games as abm_b64k, so the
#           comparison is at equal data. abk_1v1 itself is the equal-updates control.
D=/kvm/polytopia-ai-data/ab; L=/kvm/polytopia-ai-data/runs
/kvm/ab2/run.sh abm_b64k --features --n-players 2:100 --rollout-steps 65536 &
/kvm/ab2/run.sh abm_u600 --features --n-players 2:100 --init $D/runs/abk_1v1/ckpt_final.pt &
wait
echo "$(date -u +%FT%TZ) abm arms done"
mkdir -p $D/eval-abm
for f in 1v1_v_ctrl ctrl_v_r77 1v1_v_r77 ctrl_crazy 1v1_crazy r77_crazy; do cp -n $D/eval-abl/$f.jsonl $D/eval-abm/; done
ENGINE_DIR=/kvm/ab2/ext AI_DIR=/kvm/ab2/ai DEVICE=cuda JOBS=3 /kvm/ab2/ai/tools/ab_eval.sh $D/eval-abm \
  r77=$L/auto_r0077/ckpt_final.pt ctrl=$D/runs/abg_ctrl/ckpt_final.pt 1v1=$D/runs/abk_1v1/ckpt_final.pt \
  u600=$D/runs/abm_u600/ckpt_final.pt b64k=$D/runs/abm_b64k/ckpt_final.pt
echo "$(date -u +%FT%TZ) eval-abm done"
