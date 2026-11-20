#!/bin/bash
# Fourth A/B on /kvm/ab2 (control abg_ctrl, same build and seed).
# abi_pool: the pool without easy/normal/hard built-in AI, set to 0 explicitly (entries left out keep their defaults).
# abi_expl: an exploiter, 1v1 against round 77 only. When it is done, abj_league trains with the exploiter in the
# pool (0.15, taken from self) beside the usual pool; then the matches.
D=/kvm/polytopia-ai-data/ab; L=/kvm/polytopia-ai-data/runs
P0="retail:easy=0,retail:normal=0,retail:hard=0,retail:crazy=0"
/kvm/ab2/run.sh abi_pool --features --pool-weights retail:easy=0,retail:normal=0,retail:hard=0,retail:crazy=0.05,bc=0.10,bpolait=0.20,self=0.65 &
/kvm/ab2/run.sh abi_expl --features --n-players 2:100 --pool-weights $P0,bc=0,bpolait=0,self=0,champ=1.0 \
  --extra-opponents bpolait=$L/bpbc_g0412/ckpt_final.pt,champ=$L/auto_r0077/ckpt_final.pt
echo "$(date -u +%FT%TZ) abi_expl exit $?"
/kvm/ab2/run.sh abj_league --features \
  --pool-weights retail:easy=0.025,retail:normal=0.075,retail:hard=0.075,retail:crazy=0.075,bc=0.10,bpolait=0.20,self=0.30,expl=0.15 \
  --extra-opponents bpolait=$L/bpbc_g0412/ckpt_final.pt,expl=$D/runs/abi_expl/ckpt_final.pt &
wait %1
echo "$(date -u +%FT%TZ) abi_pool done"
mkdir -p $D/eval-abi; for f in ctrl_v_r77 ctrl_crazy r77_crazy; do cp -n $D/eval-abg/$f.jsonl $D/eval-abi/; done
ENGINE_DIR=/kvm/ab2/ext AI_DIR=/kvm/ab2/ai DEVICE=cuda JOBS=3 /kvm/ab2/ai/tools/ab_eval.sh $D/eval-abi \
  r77=$L/auto_r0077/ckpt_final.pt ctrl=$D/runs/abg_ctrl/ckpt_final.pt pool=$D/runs/abi_pool/ckpt_final.pt \
  expl=$D/runs/abi_expl/ckpt_final.pt
wait
echo "$(date -u +%FT%TZ) abj_league done"
mkdir -p $D/eval-abj; for f in ctrl_v_r77 ctrl_crazy r77_crazy; do cp -n $D/eval-abg/$f.jsonl $D/eval-abj/; done
ENGINE_DIR=/kvm/ab2/ext AI_DIR=/kvm/ab2/ai DEVICE=cuda JOBS=3 /kvm/ab2/ai/tools/ab_eval.sh $D/eval-abj \
  r77=$L/auto_r0077/ckpt_final.pt ctrl=$D/runs/abg_ctrl/ckpt_final.pt league=$D/runs/abj_league/ckpt_final.pt \
  expl=$D/runs/abi_expl/ckpt_final.pt
