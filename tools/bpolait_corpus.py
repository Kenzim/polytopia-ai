"""Turn the headless client's finished BPolait games into training shards of BPolait's own moves, and list the games.

    python tools/bpolait_corpus.py [--games /kvm/polytopia/client/games] [--out /kvm/polytopia-ai-data/human/bpolait]

Reads every `<id>.json` record in the client's games folder whose `opponents` names `bpolait` and whose game is over,
links its two state blobs into OUT/raw, writes OUT/seats.tsv (game id and BPolait's player id) and runs
`polyai-human --seats` so only BPolait's commands become rows (`game_key = bpolait:<id>`): about one game in five
(fixed by its id) goes to OUT/val for held-out agreement, the rest to OUT/shards. Then prints one line per
game (map, size, result for us, score, the model that played our seat) and the converter's per-game status.
Rerunning rebuilds everything from the records, so it is safe to run after every batch.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path

CONVERTER = Path(__file__).resolve().parents[1] / "target/release/polyai-human"


def held_out(gid: str) -> bool:
    """About one game in five, fixed by the id."""
    return int(hashlib.sha256(gid.encode()).hexdigest(), 16) % 5 == 0


def our_model(rec: dict) -> str:
    ids = collections.Counter(
        c["policy"]["policy_id"] for c in rec.get("commands", []) if c.get("ours") and c.get("policy", {}).get("policy_id")
    )
    return ids.most_common(1)[0][0] if ids else "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", default="/kvm/polytopia/client/games")
    ap.add_argument("--out", default="/kvm/polytopia-ai-data/human/bpolait")
    a = ap.parse_args()
    games, out = Path(a.games), Path(a.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for f in raw.iterdir():
        f.unlink()
    seats, rows = [], []
    for f in sorted(games.glob("*.json")):
        rec = json.loads(f.read_text())
        bp = [int(p) for p, who in (rec.get("opponents") or {}).items() if who == "bpolait"]
        res = rec.get("result") or {}
        if not bp or not res.get("over"):
            continue
        gid = rec["game_id"]
        for key, suffix in (("initial_file", "init"), ("final_file", "cur")):
            (raw / f"{gid}_{suffix}.bin").symlink_to(games / rec[key])
        seats.append(f"{gid} {','.join(map(str, bp))}")
        scores = dict((p, s) for p, s in res.get("scores", []))
        us = rec["our_player_id"]
        rows.append((gid, rec.get("map_type_name"), rec.get("map_size"), res.get("for_us"),
                     scores.get(us), scores.get(bp[0]), res.get("turns"), our_model(rec)))
    # Held-out games are chosen by game id, so a game never changes side as the corpus grows.
    split = {"shards": [s for s in seats if not held_out(s.split()[0])], "val": [s for s in seats if held_out(s.split()[0])]}
    meta = {}
    for name, lines in split.items():
        (out / f"seats_{name}.tsv").write_text("\n".join(lines) + "\n")
        subprocess.run([str(CONVERTER), "--raw", str(raw), "--out", str(out / name), "--seats", str(out / f"seats_{name}.tsv"),
                        "--key-prefix", "bpolait"], check=True, stderr=subprocess.DEVNULL)
        meta.update({m["id"]: {**m, "split": name} for m in map(json.loads, (out / name / "meta.jsonl").read_text().splitlines())})
    total = 0
    print(f"{'game':8} {'map':12} {'size':>4} {'result':7} {'us':>6} {'bpolait':>7} {'turns':>5} {'our model':24} converter")
    for gid, mt, size, r, s_us, s_bp, turns, model in rows:
        m = meta.get(gid, {})
        total += m.get("rows", 0)
        print(f"{gid[:8]} {mt or '?':12} {size or 0:>4} {r or '?':7} {s_us or 0:>6} {s_bp or 0:>7} {turns or 0:>5} {model:24} "
              f"{m.get('split', '')} {m.get('status', 'missing')} {m.get('rows', 0)} rows {m.get('reason', '')}")
    print(f"{len(rows)} finished BPolait games ({len(split['val'])} held out in {out / 'val'}), "
          f"{total} BPolait positions; training shards in {out / 'shards'}")


if __name__ == "__main__":
    main()
