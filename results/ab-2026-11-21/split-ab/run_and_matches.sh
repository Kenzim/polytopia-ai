#!/bin/bash
# Fifth A/B on /kvm/ab2 (control abg_ctrl, same build, seed 900 unless stated).
# Why: the exploiter abi_expl (1v1 only, against round 77 only) beat every other arm (56.4% vs ctrl) while level
# with r77, and the pool arm with more self-play lost (41.0%). These arms split its two changes and measure seed noise.
# abk_ctrl2: the control with seed 901 (run-to-run noise of one A/B arm).
# abk_champ: the loop pool with self=0 and champ=0.45 (round 77 frozen) in its place.
# abk_1v1:   the loop pool, two-player games only.
# abk_expl2: abi_expl continued for 600 updates, still 1v1 against round 77 only.
D=/kvm/polytopia-ai-data/ab; L=/kvm/polytopia-ai-data/runs
BP=bpolait=$L/bpbc_g0412/ckpt_final.pt; CH=champ=$L/auto_r0077/ckpt_final.pt
P0="retail:easy=0,retail:normal=0,retail:hard=0,retail:crazy=0"
/kvm/ab2/run.sh abk_ctrl2 --features --seed 901 &
/kvm/ab2/run.sh abk_champ --features \
  --pool-weights retail:easy=0.025,retail:normal=0.075,retail:hard=0.075,retail:crazy=0.075,bc=0.10,bpolait=0.20,self=0,champ=0.45 \
  --extra-opponents $BP,$CH &
/kvm/ab2/run.sh abk_1v1 --features --n-players 2:100 &
/kvm/ab2/run.sh abk_expl2 --features --init $D/runs/abi_expl/ckpt_final.pt --updates 600 --n-players 2:100 \
  --pool-weights $P0,bc=0,bpolait=0,self=0,champ=1.0 --extra-opponents $BP,$CH &
wait
echo "$(date -u +%FT%TZ) abk arms done"
mkdir -p $D/eval-abk; for f in ctrl_v_r77 ctrl_crazy r77_crazy; do cp -n $D/eval-abg/$f.jsonl $D/eval-abk/; done
ENGINE_DIR=/kvm/ab2/ext AI_DIR=/kvm/ab2/ai DEVICE=cuda JOBS=3 /kvm/ab2/ai/tools/ab_eval.sh $D/eval-abk \
  r77=$L/auto_r0077/ckpt_final.pt ctrl=$D/runs/abg_ctrl/ckpt_final.pt ctrl2=$D/runs/abk_ctrl2/ckpt_final.pt \
  champ=$D/runs/abk_champ/ckpt_final.pt 1v1=$D/runs/abk_1v1/ckpt_final.pt expl2=$D/runs/abk_expl2/ckpt_final.pt
echo "$(date -u +%FT%TZ) eval-abk done"
