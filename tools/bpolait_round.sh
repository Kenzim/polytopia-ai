#!/bin/bash
# Train on the headless client's BPolait games: rebuild the corpus, fine-tune a BPolait imitator, then run one gated
# self-play round from the champion with the imitator as KL reference and as a pool opponent (docs/human-data.md,
# "Learning from BPolait").
#   1. tools/bpolait_corpus.py: BPolait's moves from every finished game, about 1 game in 5 held out by id.
#   2. Stops unless there are at least MIN_GAMES finished games (default 100).
#   3. Imitator bpbc_gNNNN: behaviour cloning from the human-imitation model hbc1_rl3_human on the human tournament
#      shards plus the BPolait shards repeated BP_REPEAT times (default 10), 3000 steps. Kept only if its top-1
#      agreement on the held-out BPolait games beats hbc1's by at least 1 point.
#   4. Waits for the autoloop's current round to be decided, stops the loop, and runs tools/human_round.sh:
#      learner from the champion, REF = imitator, `bc` pool weight 0.25. Gate and promotion as the loop; the loop is
#      restarted at the end. polyai.serve is never touched.
# Usage: tools/bpolait_round.sh [GATE_SEED_START]   (env: MIN_GAMES, BP_REPEAT, DRY=1 stops after step 3)
set -u
GSEED=${1:-7000000}
MIN_GAMES=${MIN_GAMES:-100} BP_REPEAT=${BP_REPEAT:-10}
D=/kvm/polytopia-ai-data
R=$D/runs B=$D/human/bpolait L=$D/autoloop
PY=/kvm/polytopia/.venv/bin/python
AI=/kvm/polytopia/ai
export TMPDIR=/kvm/tmp/tmp PYTHONPATH=$AI/python OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p $D/arena/bpolait
log() { echo "$(date -Is) $*" | tee -a $D/arena/bpolait/pipeline.log; }

nice -n 10 $PY $AI/tools/bpolait_corpus.py --out $B > $B/corpus.txt || { log "corpus failed"; exit 1; }
GAMES=$(grep -c " ok " $B/corpus.txt)
log "corpus: $(tail -1 $B/corpus.txt)"
[ "$GAMES" -ge "$MIN_GAMES" ] || { log "$GAMES clean games, need $MIN_GAMES; stopping"; exit 0; }

NAME=$(printf "bpbc_g%04d" "$GAMES")
if [ ! -f $R/$NAME/ckpt_final.pt ]; then
  BP=""
  for _ in $(seq "$BP_REPEAT"); do BP="$BP $(ls $B/shards/*.npz | tr '\n' ' ')"; done
  nice -n 10 $PY -m polyai.train_bc --name $NAME --model b6c96 --init $R/hbc1_rl3_human/ckpt_final.pt \
    --train $D/human/shards $BP --val $B/val --steps 3000 --lr 3e-5 --warmup-steps 100 --kl-weight 0 \
    --eval-every 1000 --ckpt-every 3000 --eval-batches 1000 --threads 4 --need-gb 1.5 \
    >> $R/$NAME.log 2>&1 || { log "imitator training failed"; exit 1; }
fi
AGREE=$(CUDA_VISIBLE_DEVICES= nice -n 10 $PY $AI/tools/agreement.py --shards $B/val -- \
  $R/hbc1_rl3_human/ckpt_final.pt $R/$NAME/ckpt_final.pt 2>/dev/null | grep -o "top-1 [0-9.]*" | awk '{print $2}' | paste -sd' ')
log "held-out BPolait top-1: hbc1 / $NAME = $AGREE"
read -r BASE NEW <<< "$AGREE"
python3 -c "import sys; sys.exit(0 if $NEW >= $BASE + 0.01 else 1)" || { log "imitator not better than hbc1; stopping"; exit 0; }
[ "${DRY:-0}" = 1 ] && { log "DRY=1: stopping before the self-play round"; exit 0; }

ROUND=$($PY -c "import json; print(json.load(open('$L/state.json'))['round'])")
log "waiting for autoloop round $ROUND to be decided"
until $PY -c "import json,sys; h=[json.loads(l) for l in open('$L/history.jsonl')]; sys.exit(0 if h and h[-1]['round'] >= $ROUND else 1)"; do
  sleep 60
done
systemctl stop polyai-autoloop
CHAMP=$($PY -c "import json; print(json.load(open('$L/state.json'))['champion'])")
log "loop stopped; round from $CHAMP with reference $NAME"
REF=$R/$NAME/ckpt_final.pt BC_WEIGHT=0.25 TAG=bpolait ARENA_DIR=bpolait SEED=303 \
  $AI/tools/human_round.sh "brl_${NAME#bpbc_}" "$CHAMP" "$CHAMP" "$GSEED"
