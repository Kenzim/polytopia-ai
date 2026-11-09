# Automatic training loop

`tools/autoloop.py`, run by the systemd unit `polyai-autoloop.service` on the dev host (192.168.11.60), RTX 3060.

## What a round does

1. **Train** a candidate `auto_rNNNN` from the current champion with seed NNNN, 300 updates of Might self-play.
   Recipe: the might1 config (KL 0.02 to bc3, pool of retail AIs, bc3 and own snapshots, 55/25/20 player mix)
   plus the advantage filter at 0.5 (`rec_advf`, the best recipe A/B and half the training step).
   Engine features (`features`) and the belief auxiliary (`belief_weight`) are switches in `config.json`, off:
   both beat plain controls on some seeds, but no features or belief model has beaten rl3 u480 head to head.
2. **Gate:** 400 games (200 seeds, seats swapped) head to head against the champion, Might 1v1 suite, on a fresh
   seed block per round (`2,000,000 + 1000 × round`), so the gate cannot be overfitted to fixed seeds.
3. **Guard:** 200 games against retail Crazy on fixed seeds (from 1,000,000); the champion's score on the same
   seeds is measured once per champion and cached in `champions/`.
4. **Decide:** promote if gate score ≥ 0.55 **and** Crazy score ≥ champion's − 0.08.
   0.55 over 400 games is two standard errors above an even match: a model no better than the champion passes by
   chance about 2.3% of rounds, one 2 points weaker about 0.3%. The Crazy guard (about 1.9 standard errors of the
   difference at 200 games each) stops a candidate that beats the champion only by losing general strength.
5. A candidate that is not promoted is pruned to `ckpt_final.pt` (a full run directory is about 440 MB).

A round takes about 2.5 to 5 hours depending on host load (training measured at about 20 s per update on
2026-11-09, gate and guard about 1 h), so roughly 5 to 9 rounds a day. The champion's Crazy baseline for rl3 u480
is 152-48-0 of 200, 76.0%, so a candidate needs at least 68% to pass the guard.

## Resources

* GPU: training caps itself at 1.5 GiB (`need_gb`; peak about 0.95 GB) and waits and retries if less is free.
  Evaluation runs on the CPU.
* CPU: training 12 env + 4 torch threads, evaluation 8 threads, one phase at a time. The unit sets `Nice=15`,
  `CPUWeight=20`, `CPUQuota=2000%` (20 cores at most) and idle disk priority.

## Files (`/kvm/polytopia-ai-data/autoloop/`)

| file | content |
|---|---|
| `status.txt` | round, phase, champion, last gate result |
| `loop.log` | one line per phase start, end and failure |
| `history.jsonl` | one line per decided round with gate, guard and decision |
| `promotions.jsonl` | one line per promotion |
| `state.json` | current round and champion (the loop's only mutable state) |
| `config.json` | settings, read at every start; edit and restart to change them |
| `rounds/rNNNN/` | `round.json`, `train.log`, `gate.jsonl`, `crazy.jsonl`, logs, `result.json` |

Candidate runs live in `/kvm/polytopia-ai-data/runs/auto_rNNNN/`.

## Start, stop, check

```
systemctl start polyai-autoloop        # also starts at boot (enabled)
systemctl stop polyai-autoloop         # stops the loop and its current phase; start resumes it
cat /kvm/polytopia-ai-data/autoloop/status.txt
/kvm/polytopia/.venv/bin/python /kvm/polytopia/ai/tools/autoloop.py status
journalctl -u polyai-autoloop -n 50
```

Install or update the unit: `cp ai/deploy/systemd/polyai-autoloop.service /etc/systemd/system/ && systemctl daemon-reload`.

## Crashes and restarts

Every phase can be re-run: training resumes from `ckpt_last.pt` (written every 20 updates), and the gate and guard
skip finished games. The decision is written to `result.json` before `state.json` changes the champion. A failing
phase is retried after 10 minutes, doubling to 1 hour; after 3 failures the loop notifies the AI manager chat.
systemd restarts the loop itself after 2 minutes if it dies.

## Promotions

The loop never changes the model `polyai.serve` uses. On a promotion it appends to `promotions.jsonl` and sends a
Forge notification (token in `/root/.config/polytopia/autoloop_notify_token`, owner-only, never in the repo),
which wakes the AI manager chat to record it in the ledger and report it. The client manager decides whether to
serve the new champion.

## Dry run

`autoloop.py run --dir /kvm/tmp/autoloop-dry --dry-run --rounds 1` runs one tiny round (2 updates, 8-game gate and
guard) in a scratch directory. Tested 2026-11-09: a full round, a kill at update 25 of 40 with resume from update
20, and the not-promoted path with pruning.
