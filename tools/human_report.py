#!/usr/bin/env python3
"""Check the human-game shards and summarise the per-game meta file (docs/human-data.md).

usage: human_report.py [--dir /kvm/polytopia-ai-data/human]

Loads every shards/human-*.npz with polyai.data.Shard and checks that source is 3 for every row, every game_key starts
with `human:`, `chosen` lies inside its own legal list, winner is -1 exactly where capped is 1, and the keys of
the games in the shards are the ok games of meta.jsonl. Then prints the counts by status, mode, size, players and tribe.
Exit status 1 on any failed check.
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from polyai.data import Shard  # noqa: E402

MODES = {1: "perfection", 2: "domination", 3: "glory", 4: "might", 5: "custom"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/kvm/polytopia-ai-data/human")
    args = ap.parse_args()
    bad = []
    keys = set()
    rows = 0
    for path in sorted(glob.glob(os.path.join(args.dir, "shards", "human-*.npz"))):
        s = Shard.load(path)
        a = s.a
        n = len(s)
        rows += n
        off, chosen = a["act_off"], a["chosen"]
        sizes = off[1:] - off[:-1]
        if not ((chosen >= 0) & (chosen < sizes)).all():
            bad.append(f"{path}: chosen outside its legal list")
        if not (a["source"] == 3).all():
            bad.append(f"{path}: source is not 3 everywhere")
        gk = [str(k) for k in s.game_key]
        if not all(k.startswith("human:") for k in gk):
            bad.append(f"{path}: a game_key without the human: prefix")
        keys.update(gk)
        if not ((a["winner"] < 0) == (a["capped"] == 1)).all():
            bad.append(f"{path}: winner and capped disagree")
        if not (a["ours"] == 0).all():
            bad.append(f"{path}: ours is set")
        print(f"{os.path.basename(path)}: {n} positions, {len(set(gk))} games, legal list mean {sizes.mean():.1f}")
    meta = [json.loads(line) for line in open(os.path.join(args.dir, "meta.jsonl"))]
    ok = {"human:" + m["id"] for m in meta if m["status"] == "ok" and m["rows"] > 0}
    if keys != ok:
        bad.append(f"shard games {len(keys)} differ from ok games with rows {len(ok)}")
    by_status = collections.Counter(m["status"] for m in meta)
    print(f"\ngames in meta: {len(meta)}  positions: {rows}")
    for k, v in sorted(by_status.items()):
        print(f"  {k}: {v}")
    good = [m for m in meta if m["status"] == "ok"]
    print(f"illegal commands in ok games: {sum(m['illegal'] for m in good)}")
    print("mode:", dict(collections.Counter(MODES.get(m["mode"], m["mode"]) for m in good)))
    print("size:", dict(collections.Counter(f"{m['height']}x{m['width']}" for m in good)))
    print("players:", dict(sorted(collections.Counter(m["players"] for m in good).items())))
    print("humans:", dict(sorted(collections.Counter(m["humans"] for m in good).items())))
    print("tribes:", dict(sorted(collections.Counter(t for m in good for t in m["tribes"]).items())))
    print("finished (engine over):", sum(m["over"] for m in good), "of", len(good), "; with a winner:", sum(bool(m["winners"]) for m in good))
    for b in bad:
        print("FAIL", b)
    print("OK" if not bad else f"{len(bad)} check(s) failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
