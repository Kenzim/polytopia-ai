#!/usr/bin/env python3
"""Report on polyai-arena JSONL results: per-pair win rates and a Bradley-Terry Elo fit.

    /kvm/polytopia/.venv/bin/python ai/tools/elo.py FILE.jsonl [FILE2.jsonl ...] [--anchor retail:normal]
        [--anchor-elo 1000] [--bootstrap 300] [--seed 0] [--markdown]

Per pair: games, wins for each side, draws, win rate of the first agent (a draw counts as half a win) with a Wilson
95% interval, the mean score margin (first minus second), the share of games adjudicated at the turn limit, and the
mean wall time.

Elo: Bradley-Terry strengths fitted by the MM iteration over all games, a draw counting as half a win each way, plus a
weak prior of one drawn game in total between every pair of agents so that an unbeaten agent gets a finite rating.
Elo = 1000 + 400 log10(strength / strength of the anchor). The intervals are percentiles of a bootstrap that resamples
whole seeds (both seatings of a seed stay together), so they include map and tribe variance.
Seats are not modelled: every pair plays both seatings of every seed.
"""

import argparse
import json
import math
import sys
from collections import defaultdict

import numpy as np

Z95 = 1.959963984540054


def load_games(paths):
    """Games as dicts with the first/second agent in seat order, winner as 0, 1 or None (draw)."""
    games = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                a, b = r["agents"]
                w = r["winner"]
                games.append(
                    {
                        "seed": r["seed"],
                        "a": a,
                        "b": b,
                        "winner": w if w in (0, 1) else None,
                        "margin": r["scores"][0] - r["scores"][1],
                        "adjudicated": bool(r["adjudicated"]),
                        "wall_ms": r.get("wall_ms", 0.0),
                    }
                )
    return games


def wilson(p_wins, n, z=Z95):
    """Wilson score interval for `p_wins` successes (may be fractional, draws as halves) out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = p_wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pair_stats(games):
    """Per unordered pair (names sorted): a dict of counts and means, with `a` the alphabetically first agent."""
    acc = defaultdict(lambda: {"n": 0, "wa": 0, "wb": 0, "d": 0, "margin": 0.0, "adj": 0, "wall": 0.0})
    for g in games:
        flip = g["a"] > g["b"]
        a, b = (g["b"], g["a"]) if flip else (g["a"], g["b"])
        s = acc[(a, b)]
        s["n"] += 1
        w = g["winner"]
        if w is None:
            s["d"] += 1
        elif (w == 0) != flip:
            s["wa"] += 1
        else:
            s["wb"] += 1
        s["margin"] += -g["margin"] if flip else g["margin"]
        s["adj"] += g["adjudicated"]
        s["wall"] += g["wall_ms"]
    out = {}
    for key, s in acc.items():
        n = s["n"]
        score = s["wa"] + 0.5 * s["d"]
        lo, hi = wilson(score, n)
        out[key] = {
            "n": n,
            "wins_a": s["wa"],
            "wins_b": s["wb"],
            "draws": s["d"],
            "rate": score / n,
            "lo": lo,
            "hi": hi,
            "margin": s["margin"] / n,
            "adjudicated": s["adj"] / n,
            "wall_s": s["wall"] / n / 1000.0,
        }
    return out


def _matrices(games, agents):
    idx = {a: i for i, a in enumerate(agents)}
    k = len(agents)
    wins = np.zeros((k, k))  # wins[i, j]: score of i against j (draw = 0.5)
    for g in games:
        i, j = idx[g["a"]], idx[g["b"]]
        w = g["winner"]
        if w is None:
            wins[i, j] += 0.5
            wins[j, i] += 0.5
        elif w == 0:
            wins[i, j] += 1
        else:
            wins[j, i] += 1
    return wins


def fit_bt(wins, prior=1.0, iters=2000, tol=1e-10):
    """Bradley-Terry log-strengths from a score matrix, by the MM algorithm. `prior` drawn games per pair are added."""
    k = wins.shape[0]
    w = wins.copy()
    if k > 1:
        w += prior / 2.0 * (1 - np.eye(k))
    n = w + w.T
    wi = w.sum(axis=1)
    g = np.ones(k)
    for _ in range(iters):
        denom = (n / (g[:, None] + g[None, :])).sum(axis=1)
        new = wi / denom
        new /= np.exp(np.log(new).mean())
        if np.max(np.abs(np.log(new) - np.log(g))) < tol:
            g = new
            break
        g = new
    return np.log(g)


def to_elo(logg, anchor_idx, anchor_elo):
    e = 400.0 * logg / math.log(10)
    return anchor_elo + e - e[anchor_idx]


def elo_table(games, anchor="retail:normal", anchor_elo=1000.0, bootstrap=300, seed=0):
    """Elo per agent with a bootstrap 95% interval. Returns (agents, elo, lo, hi, anchor_used)."""
    agents = sorted({g["a"] for g in games} | {g["b"] for g in games})
    if anchor not in agents:
        anchor = agents[0]
    ai = agents.index(anchor)
    elo = to_elo(fit_bt(_matrices(games, agents)), ai, anchor_elo)
    lo = hi = elo.copy()
    if bootstrap > 0:
        by_seed = defaultdict(list)
        for g in games:
            by_seed[g["seed"]].append(g)
        seeds = sorted(by_seed)
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(bootstrap):
            pick = rng.integers(0, len(seeds), len(seeds))
            sample = [g for s in pick for g in by_seed[seeds[s]]]
            samples.append(to_elo(fit_bt(_matrices(sample, agents), iters=500, tol=1e-8), ai, anchor_elo))
        samples = np.array(samples)
        lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    return agents, elo, lo, hi, anchor


def format_pairs(stats, md=False):
    head = ["pair", "games", "W-L-D (first)", "win rate", "95% CI", "mean margin", "adjudicated", "s/game"]
    rows = []
    for (a, b), s in sorted(stats.items()):
        rows.append(
            [
                f"{a} vs {b}",
                str(s["n"]),
                f"{s['wins_a']}-{s['wins_b']}-{s['draws']}",
                f"{100 * s['rate']:.1f}%",
                f"{100 * s['lo']:.1f} to {100 * s['hi']:.1f}",
                f"{s['margin']:+.0f}",
                f"{100 * s['adjudicated']:.0f}%",
                f"{s['wall_s']:.1f}",
            ]
        )
    return render(head, rows, md)


def format_elo(agents, elo, lo, hi, anchor, md=False):
    order = np.argsort(-elo)
    head = ["agent", "Elo", "95% CI"]
    rows = [[agents[i] + (" (anchor)" if agents[i] == anchor else ""), f"{elo[i]:.0f}", f"{lo[i]:.0f} to {hi[i]:.0f}"] for i in order]
    return render(head, rows, md)


def render(head, rows, md):
    if md:
        lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines)
    widths = [max(len(head[c]), *(len(r[c]) for r in rows)) for c in range(len(head))]
    fmt = "  ".join(f"{{:<{w}}}" if c == 0 else f"{{:>{w}}}" for c, w in enumerate(widths))
    return "\n".join([fmt.format(*head)] + [fmt.format(*r) for r in rows])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("files", nargs="+")
    p.add_argument("--anchor", default="retail:normal")
    p.add_argument("--anchor-elo", type=float, default=1000.0)
    p.add_argument("--bootstrap", type=int, default=300, help="bootstrap resamples (0 for none)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--markdown", action="store_true")
    a = p.parse_args(argv)
    games = load_games(a.files)
    if not games:
        print("no games", file=sys.stderr)
        return 1
    seeds = {g["seed"] for g in games}
    adj = sum(g["adjudicated"] for g in games)
    print(f"{len(games)} games, {len(seeds)} seeds, {100 * adj / len(games):.1f}% adjudicated at the turn limit\n")
    print(format_pairs(pair_stats(games), a.markdown))
    print()
    agents, elo, lo, hi, anchor = elo_table(games, a.anchor, a.anchor_elo, a.bootstrap, a.seed)
    if anchor != a.anchor:
        print(f"anchor {a.anchor} not in the data; anchored {anchor} at {a.anchor_elo:.0f}")
    print(format_elo(agents, elo, lo, hi, anchor, a.markdown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
