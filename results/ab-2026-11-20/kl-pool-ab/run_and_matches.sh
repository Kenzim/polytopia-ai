#!/bin/bash
# Third A/B on /kvm/ab2 (control abg_ctrl): abh_kl0 (no KL anchor to the imitation model) runs from 12:35Z; abh_pool
# (built-in AI only as Crazy at 0.05, self-play 0.65) starts when the abg arms finish. Then the matches, reusing
# ctrl_v_r77 and the Crazy matches of ctrl and r77 from eval-abg (same seeds).
D=/kvm/polytopia-ai-data/ab; L=/kvm/polytopia-ai-data/runs
until [ -f $D/runs/abg_ctrl/ckpt_final.pt ] && [ -f $D/runs/abg_vopp/ckpt_final.pt ]; do sleep 30; done
echo "$(date -u +%FT%TZ) abh_pool start"
/kvm/ab2/run.sh abh_pool --features --pool-weights retail:crazy=0.05,bc=0.10,bpolait=0.20,self=0.65
echo "$(date -u +%FT%TZ) abh_pool exit $?"
until [ -f $D/runs/abh_kl0/ckpt_final.pt ] && grep -q "ALL DONE" $D/eval-abg/done.txt 2>/dev/null; do sleep 30; done
mkdir -p $D/eval-abh; for f in ctrl_v_r77 ctrl_crazy r77_crazy; do cp -n $D/eval-abg/$f.jsonl $D/eval-abh/; done
ENGINE_DIR=/kvm/ab2/ext AI_DIR=/kvm/ab2/ai DEVICE=cuda JOBS=3 /kvm/ab2/ai/tools/ab_eval.sh $D/eval-abh \
  r77=$L/auto_r0077/ckpt_final.pt ctrl=$D/runs/abg_ctrl/ckpt_final.pt kl0=$D/runs/abh_kl0/ckpt_final.pt \
  pool=$D/runs/abh_pool/ckpt_final.pt
