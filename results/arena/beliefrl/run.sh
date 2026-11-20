set -x
export TMPDIR=/kvm/tmp/tmp UV_CACHE_DIR=/kvm/tmp/uv-cache
cd /kvm/polytopia/ai/python
R=/kvm/polytopia-ai-data/runs; O=/kvm/polytopia-ai-data/arena/beliefrl
A=$R/beliefrl_b6c96/ckpt_final.pt; B=$R/might1_b6c96/ckpt_u00300.pt
PY="nice -n 10 /kvm/polytopia/.venv/bin/python"
$PY -m polyai.h2h --a $A --b $B --seeds 200 --mode might --threads 8 --device cpu --out $O/h2h.jsonl
$PY -m polyai.arena --model $A --opponent retail:hard,retail:crazy --seeds 200 --mode might --threads 8 --device cpu --out $O/belief_vs_retail.jsonl
$PY -m polyai.belief --model $A --shards $(ls -d /kvm/polytopia-ai-data/bc-might-truth-split/val/* 2>/dev/null | tr '\n' ' ') --device cpu --out $O/belief_u300.json
echo ALLDONE
