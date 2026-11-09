#!/bin/bash
# One self-play fine-tune round from the human-BC model, with the autoloop's exact recipe and gate
# (docs/autoloop.md), run outside the loop while the loop is paused. Waits for the loop to be stopped,
# trains, gates against the champion, runs the Crazy guard, then sends a Forge notification.
# Usage: tools/human_round.sh NAME INIT_CKPT CHAMPION_CKPT GATE_SEED_START
set -u
NAME=$1 INIT=$2 CHAMP=$3 GSEED=$4
D=/kvm/polytopia-ai-data
R=$D/runs
A=$D/arena/human
PY=/kvm/polytopia/.venv/bin/python
mkdir -p $A
export TMPDIR=/kvm/tmp/tmp PYTHONPATH=/kvm/polytopia/ai/python OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /kvm/polytopia/ai/python || exit 1
log() { echo "$(date -Is) $*" | tee -a $A/$NAME.pipeline.log; }

while systemctl is-active --quiet polyai-autoloop; do sleep 30; done
log "loop inactive; training $NAME from $INIT"

while pgrep -f "polyai.rl --name $NAME " > /dev/null; do sleep 30; done  # a previous wrapper's run
RESUME=""
[ -f $R/$NAME/ckpt_last.pt ] && RESUME=--resume
if [ ! -f $R/$NAME/ckpt_final.pt ]; then
  nice -n 10 ionice -c3 $PY -m polyai.rl --name $NAME --init $INIT --ref $INIT --runs-dir $R \
    --updates 300 --games 384 --rollout-steps 32768 --n-players 2:55,3:25,4:20 --turn-limit 150 --mode might \
    --pool-weights retail:easy=0.05,retail:normal=0.15,retail:hard=0.15,retail:crazy=0.15,bc=0.15,self=0.35 \
    --lr 3e-05 --ent-coef 0.0005 --kl-coef 0.02 --kl-coef-end 0.02 --critic-warmup 2 --snapshot-every 10 \
    --max-snapshots 6 --ckpt-every 20 --adv-filter 0.5 --seed 101 --threads 12 --torch-threads 4 \
    --need-gb 1.5 --device cuda $RESUME >> $R/$NAME.log 2>&1 || { log "train failed"; exit 1; }
fi
log "trained; gate vs champion"
nice -n 10 $PY -m polyai.h2h --a $R/$NAME/ckpt_final.pt --b $CHAMP --seeds 200 --seed-start $GSEED \
  --mode might --threads 8 --device cpu --out $A/${NAME}_gate.jsonl >> $A/${NAME}_gate.log 2>&1 &
nice -n 10 $PY -m polyai.arena --model $R/$NAME/ckpt_final.pt --opponent retail:crazy --seeds 100 \
  --seed-start 1000000 --mode might --threads 8 --device cpu --out $A/${NAME}_crazy.jsonl >> $A/${NAME}_crazy.log 2>&1 &
wait
# Same promotion rule as the loop: gate >= 55% and Crazy no more than 8 points below the champion's
# cached guard score. On a pass the loop's champion is switched (recorded in promotions.jsonl); then the
# loop is restarted either way. polyai.serve is never touched.
SUM=$($PY - <<EOF
import json, sys; sys.path.insert(0, "/kvm/polytopia/ai/tools")
from pathlib import Path
from autoloop import score_file, fmt, now, write_json
L = Path("$D/autoloop"); st = json.loads((L / "state.json").read_text()); cfg = json.loads((L / "config.json").read_text())
g = score_file(Path("$A/${NAME}_gate.jsonl")); c = score_file(Path("$A/${NAME}_crazy.jsonl"))
safe = st["champion_id"].replace("/", "_").replace("@", "_u")
cc = score_file(L / "champions" / f"{safe}_crazy.jsonl")
ok = g["score"] >= cfg["gate_score"] and c["score"] >= cc["score"] - cfg["crazy_drop"]
cand = "$R/$NAME/ckpt_final.pt"; cid = "${NAME}@300"
if ok and st["champion"] == "$CHAMP":
    with open(L / "promotions.jsonl", "a") as f:
        f.write(json.dumps({"round": "human", "at": now(), "from": st["champion_id"], "to": cid, "path": cand,
                            "gate": g, "crazy": c}) + "\n")
    write_json(L / "state.json", dict(st, champion=cand, champion_id=cid, champion_since=now()))
print(f"gate {fmt(g)}; crazy {fmt(c)} (champion {100 * cc['score']:.1f}%); {'PROMOTED' if ok else 'not promoted'}")
EOF
)
log "done: $SUM"
systemctl start polyai-autoloop && log "autoloop restarted"
