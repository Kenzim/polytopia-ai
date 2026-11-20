"""Calibration of win estimates on finished games: of the sides a model gives probability p, about p should win.

usage: python -m polyai.calibrate --shards DIR [DIR ...] --model NAME=CKPT [NAME=CKPT ...] [--out FILE.json]
                                  [--players 2] [--turns 2,5,10,15,20] [--device cpu] [--batch 1024]

Shards need a `game_key` and a winner; those of `polyai-human --resign-wins` are the intended input. A checkpoint may be
a policy network (its value head is read) or a privileged critic (`critic_final.pt` of an RL run with `--priv-critic`);
a critic needs shards with the truth keys (`polyai-human --truth`). Every model is run on the same positions.

A sample is one side of one game in one window of game turns: the mean, over that side's positions in the window, of
the win probability the model gives the mover, with the label "that side won". The sides are "the winner" and "the
others", so a two-player game gives two samples per window. Games without a winner are left out. `--players N` keeps
only the games with N players (0: all), read from the `meta.jsonl` that `polyai-human` writes next to its shards.

Per model and window the report holds the Brier score, the log loss, the accuracy, the expected calibration error and,
per probability bucket, the number of samples, the mean estimate and the share that won with its 95% Wilson interval.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import schema as S
from .model import PolyNet, critic_value
from .train_bc import expand_shards, load_checkpoint

EDGES = tuple(round(0.1 * k, 1) for k in range(11))
TRUTH_KEYS = ("truth", "truth_mask", "truth_g")
FEAT_KEYS = ("f_tiles", "f_globals")


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def reliability(p: np.ndarray, y: np.ndarray, edges: tuple = EDGES) -> dict:
    """`p` float[n]: estimated probabilities, `y` [n]: 1 where the event happened. Bucket k holds edges[k] <= p <
    edges[k + 1] (the last one includes its upper edge)."""
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    n = len(p)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    buckets, ece = [], 0.0
    for k in range(len(edges) - 1):
        sel = which == k
        m = int(sel.sum())
        b = {"from": edges[k], "to": edges[k + 1], "n": m}
        if m:
            wins = int(y[sel].sum())
            lo, hi = wilson(wins, m)
            b.update(mean_p=float(p[sel].mean()), won=wins / m, lo=lo, hi=hi)
            ece += m / n * abs(b["mean_p"] - b["won"])
        buckets.append(b)
    if not n:
        return {"n": 0, "buckets": buckets}
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {"n": n, "brier": float(((p - y) ** 2).mean()),
            "log_loss": float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()),
            "accuracy": float(((p > 0.5) == (y > 0.5)).mean()), "ece": float(ece), "buckets": buckets}


@torch.no_grad()
def win_probabilities(model: PolyNet, a: dict, rows: np.ndarray, device: torch.device, batch: int = 1024) -> np.ndarray:
    """The model's win probability for the mover at positions `rows` of the shard arrays `a`. `batch` is the number of
    positions per forward pass on boards of up to 256 tiles; larger boards get proportionally fewer."""
    need = (TRUTH_KEYS if model.cfg.truth_input else ()) + (FEAT_KEYS if model.cfg.features else ())
    missing = [k for k in need if k not in a]
    if missing:
        raise SystemExit(f"the model needs shard keys {missing} (polyai-human --truth for a critic, "
                         "polyai-gen --features for engine features)")
    out = np.empty(len(rows), np.float32)
    h, w = a["tiles"].shape[1:3]
    batch = max(1, batch * 256 // max(256, h * w))
    for i in range(0, len(rows), batch):
        sel = rows[i:i + batch]
        t = {k: torch.from_numpy(np.ascontiguousarray(a[k][sel])).to(device) for k in ("tiles", "globals", *need)}
        wl = critic_value(model, t["tiles"], t["globals"], t.get("truth"), t.get("truth_mask"), t.get("truth_g"),
                          f_tiles=t.get("f_tiles"), f_globals=t.get("f_globals"))
        out[i:i + batch] = F.softmax(wl.float(), dim=-1)[:, 0].cpu().numpy()
    return out


def games_with_players(shard_dir: Path, players: int) -> set[str] | None:
    """The ids of the games with `players` players, from the meta file of a shard folder (None: every game)."""
    if players <= 0:
        return None
    meta = shard_dir / "meta.jsonl"
    if not meta.exists():
        raise SystemExit(f"--players {players} needs {meta} (written by polyai-human); pass --players 0 to keep all games")
    with open(meta) as f:
        lines = [json.loads(line) for line in f]
    return {m["id"] for m in lines if m["players"] == players}


def collect(paths: list[Path], models: dict[str, PolyNet], players: int, device: torch.device, batch: int = 1024,
            log=print) -> dict:
    """Run every model on the positions of the decided games. Returns arrays over those positions: `game` (an index),
    `won` (1 when the mover's side won), `turn` and `p` (one array per model)."""
    need = {"tiles", "globals", "winner", "turn", "game_key", "version"}
    for m in models.values():
        need.update(TRUTH_KEYS if m.cfg.truth_input else ())
        need.update(FEAT_KEYS if m.cfg.features else ())
    allowed: dict[Path, set[str] | None] = {}
    index: dict[str, int] = {}
    game, won, turn, p = [], [], [], {name: [] for name in models}
    for path in paths:
        with np.load(path) as z:
            if "game_key" not in z.files or int(z["version"].reshape(-1)[0]) != S.ENCODING_VERSION:
                raise SystemExit(f"{path}: needs a game_key and encoding version {S.ENCODING_VERSION}")
            a = {k: z[k] for k in need if k in z.files}
        if path.parent not in allowed:
            allowed[path.parent] = games_with_players(path.parent, players)
        ok = allowed[path.parent]
        keys, inv = np.unique(a["game_key"], return_inverse=True)
        keep = a["winner"] >= 0
        if ok is not None:
            keep &= np.array([str(k).split(":", 1)[-1] in ok for k in keys])[inv]
        rows = np.nonzero(keep)[0]
        log(f"{path.name}: {len(rows)} of {len(keep)} positions")
        if not len(rows):
            continue
        ids = np.array([index.setdefault(str(k), len(index)) for k in keys])
        game.append(ids[inv[rows]])
        won.append((a["winner"][rows] == 0).astype(np.int64))
        turn.append(a["turn"][rows].astype(np.int64))
        for name, m in models.items():
            p[name].append(win_probabilities(m, a, rows, device, batch))
    if not game:
        raise SystemExit("no position of a decided game in the shards")
    return {"game": np.concatenate(game), "won": np.concatenate(won), "turn": np.concatenate(turn),
            "p": {name: np.concatenate(v) for name, v in p.items()}}


def window_labels(turns: list[int]) -> list[str]:
    lo = [None, *[t + 1 for t in turns]]
    hi = [*turns, None]
    return [f"<={h}" if a is None else f">={a}" if h is None else f"{a}-{h}" for a, h in zip(lo, hi)]


def report(data: dict, turns: list[int]) -> dict:
    """Per model and turn window, the `reliability` of the samples (see the module docstring). `turns` are the last
    turns of the windows; later turns form one more window."""
    labels = window_labels(turns)
    n_win = len(labels)
    win = np.digitize(data["turn"], turns, right=True)
    key = (data["game"] * 2 + data["won"]) * n_win + win
    uniq, inv = np.unique(key, return_inverse=True)
    count = np.bincount(inv)
    y, w = (uniq // n_win) % 2, uniq % n_win
    out = {}
    for name, p in data["p"].items():
        mean = np.bincount(inv, weights=p) / count
        out[name] = {labels[k]: reliability(mean[w == k], y[w == k]) for k in range(n_win)}
    return {"games": int(len(np.unique(data["game"]))), "positions": int(len(key)), "windows": labels, "models": out}


def format_report(rep: dict) -> str:
    lines = [f"{rep['games']} games, {rep['positions']} positions"]
    for name, by_window in rep["models"].items():
        for label, r in by_window.items():
            if not r["n"]:
                continue
            lines.append(f"\n{name}, turns {label}: {r['n']} samples, Brier {r['brier']:.4f}, log loss "
                         f"{r['log_loss']:.4f}, accuracy {100 * r['accuracy']:.1f}%, ECE {r['ece']:.4f}")
            for b in r["buckets"]:
                if b["n"]:
                    lines.append(f"  p {b['from']:.1f}-{b['to']:.1f}: n {b['n']:6d}  mean {b['mean_p']:.3f}  won "
                                 f"{100 * b['won']:5.1f}% [{100 * b['lo']:.1f}, {100 * b['hi']:.1f}]")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shards", nargs="+", required=True, help="shard files or folders")
    ap.add_argument("--model", nargs="+", required=True, metavar="NAME=CKPT")
    ap.add_argument("--out", help="write the report as JSON")
    ap.add_argument("--players", type=int, default=2, help="keep games with this many players (0: all)")
    ap.add_argument("--turns", default="2,5,10,15,20", help="last turn of each window")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    models = {}
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--model takes NAME=CKPT, got {spec}")
        models[name] = load_checkpoint(path).to(device).eval()
    turns = [int(t) for t in args.turns.split(",")]
    rep = report(collect(expand_shards(args.shards), models, args.players, device, args.batch), turns)
    print(format_report(rep))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
