#!/usr/bin/env bash
# Scheduled run of tools/cm_ids.py (systemd timer polyai-cm-ids.timer, see docs/human-data.md).
# Appends new tournament game ids to $DIR/cm_ids.tsv, keeps the full output per run under $DIR/cm_logs/
# and adds one summary line per run to $DIR/cm_runs.log. The refresh key is read by cm_ids.py from
# ~/.config/polytopia/challengermode_refresh_key and is never passed on the command line or logged.
set -uo pipefail

AI=/kvm/polytopia/ai
PY=/kvm/polytopia/.venv/bin/python
DIR=${CM_DIR:-/kvm/polytopia-ai-data/human}
KEY=$HOME/.config/polytopia/challengermode_refresh_key
export TMPDIR=/kvm/tmp/tmp

mkdir -p "$DIR/cm_logs"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="$DIR/cm_logs/run-$stamp.log"

count() { local n; n=$(wc -l < "$1" 2>/dev/null || echo 1); echo $((n > 0 ? n - 1 : 0)); }
ids_before=$(count "$DIR/cm_ids.tsv")
walked_before=$(count "$DIR/cm_walked.tsv")

nice -n 10 "$PY" "$AI/tools/cm_ids.py" --dir "$DIR" > "$log" 2>&1
rc=$?

ids_after=$(count "$DIR/cm_ids.tsv")
walked_after=$(count "$DIR/cm_walked.tsv")
errors=$(count "$DIR/cm_errors.tsv")
done_line=$(grep '^done:' "$log" | tail -n 1)
stopped=$(grep -c '^STOPPED:' "$log")

# Safety net: the key must never appear in a log. grep -F -f reads the key as a pattern without printing it.
leak=0
if [[ -s "$KEY" ]] && grep -qF -f "$KEY" "$log"; then
    leak=1
    : > "$log"
    echo "log emptied: it contained the refresh key" > "$log"
fi

printf '%s rc=%s tournaments_walked=%s new_ids=%s total_ids=%s open_errors=%s stopped=%s key_in_log=%s | %s | log=%s\n' \
    "$stamp" "$rc" "$((walked_after - walked_before))" "$((ids_after - ids_before))" "$ids_after" \
    "$errors" "$stopped" "$leak" "${done_line:-no done line}" "$log" >> "$DIR/cm_runs.log"

[[ $leak -eq 1 ]] && exit 3
exit $rc
