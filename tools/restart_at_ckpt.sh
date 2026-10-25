#!/usr/bin/env bash
# Restart a remote Windows RL run just after it writes its next ckpt_last.pt, so a changed .bat
# (for example a new --updates total) takes effect without losing updates.
# Usage: restart_at_ckpt.sh HOST DRIVE RUN_NAME
set -euo pipefail
h=$1; d=$2; n=$3
ps() { local enc; enc=$(iconv -f utf-8 -t utf-16le | base64 -w0); ssh -o BatchMode=yes "$h" "powershell -NoProfile -EncodedCommand $enc" 2>/dev/null | grep -v -e CLIXML -e '<Objs' || true; }
stamp() { echo "(Get-Item '$d:\\polytopia\\data\\runs\\$n\\ckpt_last.pt').LastWriteTime.Ticks" | ps | tr -d '\r\n '; }
t0=$(stamp)
echo "$(date +%T) $n waiting for a new checkpoint (ticks $t0)"
while [ "$(stamp)" = "$t0" ]; do sleep 15; done
sleep 20  # let the snapshot and ckpt_uNNNNN writes finish
ps <<EOF
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { \$_.CommandLine -like '*--name $n *' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force; "stopped \$(\$_.ProcessId)" }
Start-Sleep 5
& '$d:\\polytopia\\runs\\start.ps1' -Bat '$d:\\polytopia\\runs\\run_$n.bat'
EOF
echo "$(date +%T) $n restarted"
