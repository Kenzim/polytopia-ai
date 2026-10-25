#!/usr/bin/env bash
# Wait for a remote Windows RL run to write ckpt_final.pt, pull it (sha256 checked) and play it head to head
# against a reference checkpoint on the dev host CPU (8 threads, nice).
#   tools/finish_runs.sh HOST DRIVE RUN_NAME LOCAL_NAME REF_CKPT OUT_DIR [SEEDS] [SIZES]
# SIZES empty = the default 1v1 suite; for example large,huge,massive for the big-map suite.
set -u
h=$1 d=$2 n=$3 local=$4 ref=$5 out=$6 seeds=${7:-200} sizes=${8:-}
py=/kvm/polytopia/.venv/bin/python
export TMPDIR=/kvm/tmp/tmp
dst=/kvm/polytopia-ai-data/runs/remote_ckpts/$local.pt
rem="$d:\\polytopia\\data\\runs\\$n\\ckpt_final.pt"
ps() { local enc; enc=$(iconv -f utf-8 -t utf-16le | base64 -w0); ssh -o BatchMode=yes "$h" "powershell -NoProfile -EncodedCommand $enc" 2>/dev/null | grep -v -e CLIXML -e '<Objs'; }
running() { echo "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*--name $n *' }).Count" | ps | tr -d '\r\n '; }
echo "$(date -Is) $n: waiting for the run to finish"
while [ "$(running)" != "0" ]; do sleep 120; done
sleep 10
want=$(echo "(Get-FileHash '$rem' -Algorithm SHA256).Hash.ToLower()" | ps | tr -d '\r\n ')
echo "[Convert]::ToBase64String([IO.File]::ReadAllBytes('$rem'))" | ps | tr -d '\r\n' | base64 -d > "$dst"
got=$(sha256sum "$dst" | cut -d' ' -f1)
if [ "$want" != "$got" ]; then echo "$(date -Is) $n: hash mismatch ($want vs $got)"; exit 1; fi
echo "$(date -Is) $n: pulled $dst ($got)"
mkdir -p "$out"
cd "$(dirname "$0")/../python"
extra=()
[ -n "$sizes" ] && extra=(--sizes "$sizes")
nice -n 15 $py -m polyai.h2h --a "$dst" --b "$ref" --seeds "$seeds" "${extra[@]}" --device cpu --threads 8 \
  --out "$out/h2h_$local.jsonl" > "$out/h2h_$local.log" 2>&1
grep 'W-L-D' "$out/h2h_$local.log"
echo "$(date -Is) $n: done"
