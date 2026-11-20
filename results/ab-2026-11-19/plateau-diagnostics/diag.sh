#!/bin/bash
# Diagnostic matches for the gate plateau: when in a round the drop against round 77 appears, whether it is there
# when both sides sample their moves, and whether round 77's promotion over round 75 repeats on fresh seeds.
R=/kvm/tmp/abc/runs; L=/kvm/polytopia-ai-data/runs; OUT=/kvm/tmp/abc/diag
export OUT TMPDIR=/kvm/tmp/tmp OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=/kvm/tmp/ptc-engine2:/kvm/polytopia/ai/python
job() {
  cd /kvm/polytopia/ai/python
  nice -n 15 /kvm/polytopia/.venv/bin/python -m polyai.h2h --a "$2" --b "$3" --seeds 200 --seed-start "$4" \
    --mode might --threads 8 --device cpu --out "$OUT/$1.jsonl" >> "$OUT/$1.log" 2>&1
  echo "$(date -u +%FT%TZ) $1 exit $?" >> "$OUT/done.txt"
}
export -f job
xargs -P 6 -L 1 bash -c 'job "$@"' _ <<LIST
ctrl020_v_r77 $R/abc_ctrl/ckpt_u00020.pt $L/auto_r0077/ckpt_final.pt 3000000
ctrl060_v_r77 $R/abc_ctrl/ckpt_u00060.pt $L/auto_r0077/ckpt_final.pt 3000000
ctrl140_v_r77 $R/abc_ctrl/ckpt_u00140.pt $L/auto_r0077/ckpt_final.pt 3000000
ctrl300s_v_r77s $R/abc_ctrl/ckpt_final.pt:sample=1.0 $L/auto_r0077/ckpt_final.pt:sample=1.0 3000000
r77_v_r75 $L/auto_r0077/ckpt_final.pt $L/auto_r0075/ckpt_final.pt 3100000
r95_v_r75 $L/auto_r0095/ckpt_final.pt $L/auto_r0075/ckpt_final.pt 3100000
LIST
echo "$(date -u +%FT%TZ) DIAG DONE" >> $OUT/done.txt
