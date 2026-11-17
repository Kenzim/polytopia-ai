"""Loss autopsy: replay the headless client's finished games against a bot and compare per-turn curves of wins and
losses, to see where and why our model loses.

    python3 tools/loss_autopsy.py [--games /kvm/polytopia/client/games] [--opponent bpolait] [--out /kvm/tmp/autopsy]
                                  [--model ID] [--map NAME] [--size N] [--no-replay]

Every `<id>.json` record whose `opponents` names the opponent tag (any tag: `bpolait`, `polytopai`, ...), whose result
for us is `won` or `lost` and whose game ended by the rules is replayed with the `replay_stats` example of
polyai-core (built on first use), which prints per-turn statistics for both seats (see
`crates/polyai-core/examples/replay_stats.rs`). Replays are cached in OUT/games/<id>.jsonl, so a rerun only replays
new games (`--no-replay` uses the cache only). The model's own win estimate (the `value` of its moves) is added per
turn from the record. The report goes to stdout and the per-game table to OUT/games.tsv.

The report: win rates by model, map type and size; how losses end; per metric the mean of us minus the opponent at
fixed turns in wins and in losses, with an effect size and the first turn where losses fall clearly behind wins;
the opening (first researches, trains and builds of each side); and the model's value estimate in wins and losses.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import subprocess
from pathlib import Path

AI = Path(__file__).resolve().parents[1]
BIN = AI / "target/release/examples/replay_stats"
METRICS = ["score", "income", "stars", "cities", "city_levels", "pop", "villages", "enemy_cities", "units", "army",
           "techs", "kills", "losses", "territory", "explored", "attacks", "trains", "builds", "researches"]
CUM = {"trains", "builds", "researches", "attacks"}
TURNS = [2, 4, 6, 8, 10, 12, 15, 20, 25]


def build() -> None:
    subprocess.run(["cargo", "build", "--release", "-q", "-p", "polyai-core", "--example", "replay_stats"], cwd=AI, check=True)


def our_model(rec: dict) -> str:
    pol = rec.get("policies") or {}
    if pol:
        return max(pol.items(), key=lambda kv: kv[1])[0]
    ids = collections.Counter(c["policy"]["policy_id"] for c in rec.get("commands", []) if c.get("ours") and (c.get("policy") or {}).get("policy_id"))
    return ids.most_common(1)[0][0] if ids else "retail"


def values_by_turn(rec: dict, us: int) -> dict[int, float]:
    """Mean P(we win) of the model's moves per turn of ours (turn = number of our earlier end-turn commands)."""
    turn, per = 0, collections.defaultdict(list)
    for c in rec.get("commands", []):
        if c.get("player") != us:
            continue
        v = (c.get("policy") or {}).get("value")
        if c.get("ours") and v is not None:
            per[turn].append(v)
        if c.get("kind") == 15:
            turn += 1
    return {t: statistics.fmean(v) for t, v in per.items()}


def load_games(a) -> list[dict]:
    games, cache = Path(a.games), Path(a.out) / "games"
    cache.mkdir(parents=True, exist_ok=True)
    built = False
    out = []
    for f in sorted(games.glob("*.json")):
        rec = json.loads(f.read_text())
        opp = [int(p) for p, who in (rec.get("opponents") or {}).items() if who == a.opponent]
        res, sr = rec.get("result") or {}, rec.get("server_result") or {}
        if len(opp) != 1 or res.get("for_us") not in ("won", "lost") or sr.get("end_reason") != "rules":
            continue
        model = our_model(rec)
        if a.model and model != a.model:
            continue
        gid = rec["game_id"]
        path = cache / f"{gid}.jsonl"
        if not path.exists():
            if a.no_replay:
                continue
            if not built:
                build()
                built = True
            p = subprocess.run([str(BIN), str(games / rec["initial_file"]), str(games / rec["final_file"])],
                               capture_output=True, text=True)
            if p.returncode != 0:
                print(f"replay failed: {gid}: {p.stderr.strip().splitlines()[-1:]}")
                continue
            path.write_text(p.stdout)
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        us = rec["our_player_id"]
        g = {"id": gid, "model": model, "won": res["for_us"] == "won", "us": us, "opp": opp[0],
             "map": rec.get("map_type_name") or "?", "size": rec.get("map_size") or 0, "ended_at": rec.get("ended_at") or 0,
             "turns": res.get("turns") or 0, "value": values_by_turn(rec, us),
             "rows": {}, "events": [], "end": None}
        for x in lines:
            if x["t"] == "turn":
                g["rows"][(x["player"], x["turn"])] = x
            elif x["t"] == "ev":
                g["events"].append(x)
            else:
                g["end"] = x
        scores = dict(res.get("scores") or [])
        g["score_us"], g["score_opp"] = scores.get(us, 0), scores.get(opp[0], 0)
        out.append(g)
    return out


def row(g: dict, side: str, turn: int) -> dict | None:
    """The side's row at `turn`, or its last row when the game ended earlier (the state it was left in)."""
    p = g[side]
    r = g["rows"].get((p, turn))
    if r is None:
        last = max((t for (q, t) in g["rows"] if q == p), default=None)
        if last is None or last > turn:
            return None
        r = g["rows"][(p, last)]
    return r


def metric(g: dict, side: str, turn: int, m: str) -> float | None:
    p = g[side]
    if m in CUM:
        rows = [g["rows"][(p, t)] for t in range(turn + 1) if (p, t) in g["rows"]]
        return sum(r[m] for r in rows) if rows else None
    if m == "income":
        a, b = g["rows"].get((p, turn)), g["rows"].get((p, turn + 1))
        if a and b:
            return b["start_stars"] - a["stars"]
        r = row(g, side, turn)
        return r["income"] if r else None
    r = row(g, side, turn)
    return r[m] if r else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else float("nan")


def effect(a: list[float], b: list[float]) -> float:
    """Cohen's d of a minus b."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sa, sb = statistics.pvariance(a), statistics.pvariance(b)
    sd = math.sqrt((sa * len(a) + sb * len(b)) / (len(a) + len(b))) or 1.0
    return (statistics.fmean(a) - statistics.fmean(b)) / sd


def rate(games, key) -> list[str]:
    by = collections.defaultdict(list)
    for g in games:
        by[key(g)].append(g["won"])
    return [f"  {k}: {sum(v)}/{len(v)} won ({100 * sum(v) / len(v):.0f}%)" for k, v in sorted(by.items(), key=lambda kv: str(kv[0]))]


def report(games: list[dict], a) -> None:
    won = [g for g in games if g["won"]]
    lost = [g for g in games if not g["won"]]
    print(f"{len(games)} finished games against {a.opponent}: {len(won)} won, {len(lost)} lost")
    print("By model:", *rate(games, lambda g: g["model"]), sep="\n")
    print("By map type:", *rate(games, lambda g: g["map"]), sep="\n")
    print("By size:", *rate(games, lambda g: g["size"]), sep="\n")
    tribes = collections.Counter(g["end"]["tribes"].get(str(g["opp"]), "?") for g in games if g["end"])
    print("Opponent tribes:", dict(tribes))

    print("\nHow games end (turns, final score, our original capital):")
    for name, gs in (("won", won), ("lost", lost)):
        if not gs:
            continue
        cap_lost = sum(1 for g in gs if g["end"] and g["end"]["capital_holder"].get(str(g["us"])) != g["us"])
        opp_cap = sum(1 for g in gs if g["end"] and g["end"]["capital_holder"].get(str(g["opp"])) != g["opp"])
        print(f"  {name}: n={len(gs)}, turns median {statistics.median(g['turns'] for g in gs)} "
              f"(range {min(g['turns'] for g in gs)}-{max(g['turns'] for g in gs)}), score us {mean(g['score_us'] for g in gs):.0f} "
              f"vs opp {mean(g['score_opp'] for g in gs):.0f}, our capital taken in {cap_lost}, theirs taken in {opp_cap}")

    print(f"\nUs minus opponent, mean in wins | mean in losses | effect size (losses vs wins), by turn:")
    lead = {}
    for m in METRICS:
        cells = []
        for t in TURNS:
            dw = [metric(g, "us", t, m) - metric(g, "opp", t, m) for g in won if metric(g, "us", t, m) is not None and metric(g, "opp", t, m) is not None]
            dl = [metric(g, "us", t, m) - metric(g, "opp", t, m) for g in lost if metric(g, "us", t, m) is not None and metric(g, "opp", t, m) is not None]
            d = effect(dl, dw)
            if m not in lead and not math.isnan(d) and d <= -0.5:
                lead[m] = t
            cells.append(f"t{t} {mean(dw):+.1f}|{mean(dl):+.1f}|{d:+.2f}")
        print(f"  {m:12} " + "  ".join(cells))
    print("First turn where losses trail wins by an effect size of 0.5 or more:",
          ", ".join(f"{m} t{t}" for m, t in sorted(lead.items(), key=lambda kv: kv[1])) or "none")

    print("\nRaw means by turn (us / opponent) in losses, then wins:")
    for name, gs in (("lost", lost), ("won", won)):
        for m in ("score", "cities", "villages", "income", "stars", "army", "units", "techs", "explored"):
            cells = [f"t{t} {mean(metric(g, 'us', t, m) for g in gs):.1f}/{mean(metric(g, 'opp', t, m) for g in gs):.1f}" for t in TURNS]
            print(f"  {name:4} {m:9} " + "  ".join(cells))

    print("\nOpening, turns 0-5 (share of games):")
    for side in ("us", "opp"):
        firsts = collections.defaultdict(collections.Counter)
        for g in games:
            p = g[side]
            ev = [e for e in g["events"] if e["player"] == p and e["turn"] <= 5]
            for kind in ("research", "train", "build"):
                seq = [e["what"] for e in ev if e["kind"] == kind][:3]
                firsts[kind][" > ".join(seq) or "-"] += 1
            v = next((e["turn"] for e in g["events"] if e["player"] == p and e["kind"] == "capture" and e["what"] == "village"), None)
            firsts["first village turn"][v if v is not None else "none"] += 1
        for kind, c in firsts.items():
            top = ", ".join(f"{k} {100 * n / len(games):.0f}%" for k, n in c.most_common(4))
            print(f"  {side:3} {kind}: {top}")

    print("\nUnits trained by turn 10 and techs owned by turn 10, mean per game (us / opponent):")
    for name, gs in (("lost", lost), ("won", won)):
        for kind in ("train", "research"):
            c = {s: collections.Counter() for s in ("us", "opp")}
            for g in gs:
                for s in c:
                    c[s].update(e["what"] for e in g["events"] if e["player"] == g[s] and e["kind"] == kind and e["turn"] <= 10)
            keys = sorted(set(c["us"]) | set(c["opp"]), key=lambda k: -(c["us"][k] + c["opp"][k]))[:10]
            print(f"  {name:4} {kind:8} " + ", ".join(f"{k} {c['us'][k] / len(gs):.1f}/{c['opp'][k] / len(gs):.1f}" for k in keys))

    print("\nWin rate by who leads at turn 8 (lead / behind or level):")
    for m in ("cities", "villages", "income", "army", "units", "explored", "techs", "score", "city_levels"):
        ahead = [g["won"] for g in games if (metric(g, "us", 8, m) or 0) > (metric(g, "opp", 8, m) or 0)]
        behind = [g["won"] for g in games if (metric(g, "us", 8, m) or 0) <= (metric(g, "opp", 8, m) or 0)]
        pa = f"{100 * sum(ahead) / len(ahead):.0f}% of {len(ahead)}" if ahead else "-"
        pb = f"{100 * sum(behind) / len(behind):.0f}% of {len(behind)}" if behind else "-"
        print(f"  {m:12} ahead {pa:14} behind {pb}")
    print("Win rate by opponent tribe:", *rate([g for g in games if g["end"]], lambda g: g["end"]["tribes"].get(str(g["opp"]), "?")), sep="\n")
    first_fight = [next((t for (p, t), r in sorted(g["rows"].items(), key=lambda kv: kv[0][1]) if r["attacks"] or r["kills"] or r["losses"]), None) for g in games]
    print(f"First turn with combat: median {statistics.median([t for t in first_fight if t is not None])}")

    print("\nOur model's value estimate P(win), mean by turn, wins | losses:")
    print("  " + "  ".join(f"t{t} {mean(g['value'].get(t) for g in won):.2f}|{mean(g['value'].get(t) for g in lost):.2f}" for t in TURNS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", default="/kvm/polytopia/client/games")
    ap.add_argument("--opponent", default="bpolait")
    ap.add_argument("--out", default="/kvm/tmp/autopsy")
    ap.add_argument("--model", help="only games whose moves were mostly this model's")
    ap.add_argument("--map", help="only this map type")
    ap.add_argument("--size", type=int, help="only this map size")
    ap.add_argument("--no-replay", action="store_true", help="use cached replays only")
    a = ap.parse_args()
    games = load_games(a)
    games = [g for g in games if (not a.map or g["map"] == a.map) and (not a.size or g["size"] == a.size)]
    out = Path(a.out)
    with open(out / "games.tsv", "w") as f:
        f.write("game\tended_at\tmodel\tmap\tsize\tresult\tturns\tscore_us\tscore_opp\n")
        for g in games:
            f.write(f"{g['id']}\t{g['ended_at']}\t{g['model']}\t{g['map']}\t{g['size']}\t{'won' if g['won'] else 'lost'}\t"
                    f"{g['turns']}\t{g['score_us']}\t{g['score_opp']}\n")
    if not games:
        print("no games")
        return
    report(games, a)


if __name__ == "__main__":
    main()
