#!/usr/bin/env python3
"""Per-cell arena report and paired sign tests (research.md recommendation 10).

    /kvm/polytopia/.venv/bin/python ai/tools/cells.py FILE.jsonl [FILE2.jsonl ...] [--agent SUBSTR] [--min-games 8]
        [--markdown]
    /kvm/polytopia/.venv/bin/python ai/tools/cells.py --paired A.jsonl B.jsonl [--agent-a SUBSTR] [--agent-b SUBSTR]

Report mode: for the agent whose name contains --agent (default: the first `model:` agent found), its score (win 1,
draw 0.5, loss 0) per opponent, and within each opponent by our tribe, opponent tribe, our seat, map size, map preset
and tribe pair, each with a Wilson 95% interval. Cells whose interval lies wholly below the opponent's pooled score
are flagged `weak`; pairs with fewer than --min-games games are left out of the pair table.

Paired mode: two runs of the same suite (same seeds, swaps and opponent) with different models. Games are matched on
(opponent, seed, swap); the sign test counts the matched games where A scored more than B and the reverse, and gives
a two-sided exact binomial p-value over the games that differ. Identical maps and tribes cancel most of the variance,
so this reads smaller differences than two independent 400-game scores.
"""

import argparse
import json
import math
import sys
from collections import defaultdict

Z95 = 1.959963984540054
TRIBE_NAMES = {2: "Ai-Mo", 4: "Bardur", 6: "Hoodrick", 7: "Imperius", 8: "Kickoo", 9: "Luxidoor", 10: "Oumaji",
               11: "Quetzali", 12: "Vengir", 13: "Xin-xi", 14: "Yadakk", 15: "Zebasi", 17: "Cymanti"}
SIZE_NAMES = {1: "tiny", 2: "small", 3: "normal", 4: "large", 5: "huge", 6: "massive"}
PRESET_NAMES = {1: "dryland", 2: "lakes", 3: "continents", 4: "archipelago", 5: "waterworld", 6: "pangea"}


def wilson(score: float, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = score / n
    d = 1 + Z95 * Z95 / n
    c = (p + Z95 * Z95 / (2 * n)) / d
    h = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def read_lines(paths):
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def pick_agent(games: list[dict], substr: str | None) -> str:
    names = {a for g in games for a in g["agents"]}
    if substr:
        hits = sorted(n for n in names if substr in n)
    else:
        hits = sorted(n for n in names if n.startswith("model:"))
    if len(hits) != 1:
        raise SystemExit(f"--agent must pick exactly one agent; candidates {sorted(names)}, matched {hits}")
    return hits[0]


def outcome(g: dict, agent: str) -> dict | None:
    """The game from `agent`'s side, or None if it did not play."""
    if agent not in g["agents"]:
        return None
    seat = g["agents"].index(agent)
    opp = g["agents"][1 - seat]
    w = g.get("winner")
    if g.get("capped") or (g.get("capped") is None and g.get("adjudicated")):
        w = None
    score = 0.5 if w is None or w == -1 else (1.0 if w == seat else 0.0)
    setup = g.get("setup", {})
    return {"opp": opp, "seed": g["seed"], "swap": g.get("swap", 0), "seat": seat, "score": score,
            "tribe": g["tribes"][seat], "opp_tribe": g["tribes"][1 - seat],
            "size": setup.get("size"), "preset": setup.get("preset"), "turns": g.get("turns")}


def table(rows: list[dict], key, label: str, pooled: float, min_games: int, md: bool) -> list[str]:
    cells = defaultdict(lambda: [0.0, 0])
    for r in rows:
        c = cells[key(r)]
        c[0] += r["score"]
        c[1] += 1
    out = []
    for k, (s, n) in sorted(cells.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        if n < min_games:
            continue
        lo, hi = wilson(s, n)
        flag = "weak" if hi < pooled else ""
        if md:
            out.append(f"| {label} | {k} | {n} | {100 * s / n:.1f}% | {100 * lo:.1f} to {100 * hi:.1f} | {flag} |")
        else:
            out.append(f"  {label:<10} {str(k):<26} n={n:<4} {100 * s / n:5.1f}%  [{100 * lo:5.1f}, {100 * hi:5.1f}] {flag}")
    return out


def report(paths, agent_sub, min_games, md) -> None:
    games = list(read_lines(paths))
    agent = pick_agent(games, agent_sub)
    rows = [o for g in games if (o := outcome(g, agent))]
    by_opp = defaultdict(list)
    for r in rows:
        by_opp[r["opp"]].append(r)
    print(f"agent {agent}: {len(rows)} games")
    for opp, rs in sorted(by_opp.items()):
        s = sum(r["score"] for r in rs)
        pooled = s / len(rs)
        lo, hi = wilson(s, len(rs))
        print(f"\nvs {opp}: n={len(rs)} score {100 * pooled:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]")
        if md:
            print("\n| split | cell | games | score | 95% CI | |\n|---|---|---|---|---|---|")
        lines = []
        lines += table(rs, lambda r: TRIBE_NAMES.get(r["tribe"], r["tribe"]), "our tribe", pooled, 1, md)
        lines += table(rs, lambda r: TRIBE_NAMES.get(r["opp_tribe"], r["opp_tribe"]), "opp tribe", pooled, 1, md)
        lines += table(rs, lambda r: f"seat {r['seat']}", "seat", pooled, 1, md)
        lines += table(rs, lambda r: SIZE_NAMES.get(r["size"], r["size"]), "size", pooled, 1, md)
        lines += table(rs, lambda r: PRESET_NAMES.get(r["preset"], r["preset"]), "preset", pooled, 1, md)
        lines += table(rs, lambda r: f"{TRIBE_NAMES.get(r['tribe'])} v {TRIBE_NAMES.get(r['opp_tribe'])}", "pair",
                       pooled, min_games, md)
        print("\n".join(lines))


def sign_test(a_better: int, b_better: int) -> float:
    """Two-sided exact binomial p-value for a_better successes in a_better + b_better trials at p = 0.5."""
    n = a_better + b_better
    if n == 0:
        return 1.0
    k = min(a_better, b_better)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(path_a, path_b, sub_a, sub_b) -> None:
    ga, gb = list(read_lines([path_a])), list(read_lines([path_b]))
    agent_a, agent_b = pick_agent(ga, sub_a), pick_agent(gb, sub_b)
    ka = {(o["opp"], o["seed"], o["swap"]): o for g in ga if (o := outcome(g, agent_a))}
    kb = {(o["opp"], o["seed"], o["swap"]): o for g in gb if (o := outcome(g, agent_b))}
    common = sorted(set(ka) & set(kb))
    if not common:
        raise SystemExit("no matched games: the two files must share opponent, seed and swap")
    by_opp = defaultdict(list)
    for k in common:
        by_opp[k[0]].append((ka[k]["score"], kb[k]["score"]))
    print(f"A = {agent_a} ({path_a})\nB = {agent_b} ({path_b})")
    for opp, pairs in sorted(by_opp.items()):
        sa = sum(a for a, _ in pairs) / len(pairs)
        sb = sum(b for _, b in pairs) / len(pairs)
        a_b = sum(1 for a, b in pairs if a > b)
        b_b = sum(1 for a, b in pairs if b > a)
        print(f"vs {opp}: {len(pairs)} matched games, A {100 * sa:.1f}%, B {100 * sb:.1f}%, "
              f"A better {a_b}, B better {b_b}, same {len(pairs) - a_b - b_b}, sign test p = {sign_test(a_b, b_b):.3f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--agent")
    ap.add_argument("--min-games", type=int, default=8)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--paired", nargs=2, metavar=("A", "B"))
    ap.add_argument("--agent-a")
    ap.add_argument("--agent-b")
    args = ap.parse_args(argv)
    if args.paired:
        paired(args.paired[0], args.paired[1], args.agent_a, args.agent_b)
    elif args.files:
        report(args.files, args.agent, args.min_games, args.markdown)
    else:
        ap.error("give JSONL files or --paired A B")


if __name__ == "__main__":
    main(sys.argv[1:])
