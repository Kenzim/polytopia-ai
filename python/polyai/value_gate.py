"""The calibration gate of the analysis win chance (docs/analysis.md): of the players a model gives p, about p should
win, checked on held-out games per game kind and stage.

usage: python -m polyai.value_gate --rows DIR [DIR ...] (--policy NAME=CKPT | --value NAME=CKPT) [...]
                                   [--split test] [--device cuda] [--out FILE.json] [--md FILE.md]

`--rows` are folders of value shards (`value-*.npz`, `polyai-human --value-rows` or `polyai-webgen`). Rows are split
by game, never by position: a hash of the game key puts one game in ten in `test`, one in ten in `fit` (where the
calibration layer is fitted) and the rest in `train`. Rows of undecided games are left out.

A `--policy` checkpoint is gated on its raw value head (slot 0 of the winner softmax, no calibration); a `--value`
checkpoint (`polyai.train_value`) with its calibration layer, and a (kind, stage) it has no temperature for counts as
"no number given" (reported, not scored).

Per (kind, stage) the gate reports rows, games, the base rate, the Brier score against the Brier score of always
saying the base rate, the expected calibration error over ten buckets, and the worst bucket. It passes when it has at
least `MIN_GAMES` games, ECE <= 0.05, Brier <= the base rate's Brier, and no bucket holding at least 2% (and 200) of
its rows is off by more than 0.10.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .analysis import STAGE_NAMES, STAGES, Calibration, ValueNet, kind_of, load_value, policy_value_net, stage_of
from .calibrate import reliability

SPLITS = {"train": 0, "fit": 1, "test": 2}
MIN_GAMES = 100
MAX_ECE = 0.05
MAX_GAP = 0.10


def split_of_key(key: str) -> int:
    h = int.from_bytes(hashlib.sha1(key.encode()).digest()[:4], "little") % 10
    return SPLITS["test"] if h == 0 else SPLITS["fit"] if h == 1 else SPLITS["train"]


def load_rows(dirs: list[str | Path], split: str | None = None, decided: bool = True) -> list[dict]:
    """The value shards under `dirs` as dicts of arrays, keeping only the rows of `split` (None: all) and, with
    `decided`, of games with a winner. Adds `split` (int8) and `kind` (str) per row."""
    return load_splits(dirs, (split,), decided)[split]


def parse_keep(specs: list[str] | None) -> dict[str, float]:
    """`["web=0.2", "human=1"]` -> {game key prefix: share of rows kept}."""
    out = {}
    for s in specs or []:
        k, v = s.split("=", 1)
        out[k] = float(v)
    return out


def load_splits(dirs: list[str | Path], splits: tuple, decided: bool = True,
                keep: dict[str | None, dict[str, float]] | None = None) -> dict:
    """`load_rows` for several splits (None: all rows) in one pass over the files: {split: [shard dicts]}.
    `keep[split]` maps a game key prefix (before the `:`) to the share of its rows kept, drawn at random per row
    (the same rows on every load)."""
    out: dict = {s: [] for s in splits}
    files = sorted(f for d in dirs for f in Path(d).rglob("value-*.npz"))
    if not files:
        raise SystemExit(f"no value-*.npz under {dirs}")
    keep = keep or {}

    def one(f: Path) -> dict:
        got: dict = {}
        with np.load(f) as z:
            keys = z["game_key"]
            winner = z["winner"]
            sp = np.array([split_of_key(k) for k in keys.tolist()], np.int8)
            draw = np.random.default_rng(int.from_bytes(hashlib.sha1(f.name.encode()).digest()[:4], "little")).random(len(keys))
            prefix = np.char.partition(keys.astype(str), ":")[:, 0]
            sels = {}
            for split in splits:
                sel = np.ones(len(keys), bool)
                if split is not None:
                    sel &= sp == SPLITS[split]
                if decided:
                    sel &= winner >= 0
                for pre, frac in keep.get(split, {}).items():
                    sel &= ~((prefix == pre) & (draw >= frac))
                if sel.any():
                    sels[split] = sel
            if not sels:
                return got
            arrays = {k: z[k] for k in z.files if k != "version"}
        kind = kinds_of(arrays["mode"], arrays["n_players"], arrays["slot_kind"])
        for split, sel in sels.items():
            a = {k: v[sel] for k, v in arrays.items()}
            a["split"], a["kind"] = sp[sel], kind[sel]
            got[split] = a
        return got

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=12) as pool:
        for got in pool.map(one, files):
            for split, a in got.items():
                out[split].append(a)
    return out


def kinds_of(mode: np.ndarray, n_players: np.ndarray, slot_kind: np.ndarray) -> np.ndarray:
    """`kind_of` per row, computed once per distinct (mode, players, slot kinds)."""
    key = np.concatenate([mode[:, None], n_players[:, None], slot_kind], axis=1).astype(np.int16)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    names = np.array([kind_of(int(u[0]), int(u[1]), u[2:]) for u in uniq])
    return names[inv.reshape(-1)]


@torch.no_grad()
def predict_logits(net: ValueNet, shards: list[dict], device: torch.device, batch: int = 1024) -> np.ndarray:
    """Winner logits [N, 9] of every row of `shards`, in order."""
    net.eval()
    outs = []
    for a in shards:
        h, w = a["tiles"].shape[1:3]
        b = max(1, batch * 256 // max(256, h * w))
        for i in range(0, len(a["winner"]), b):
            t = lambda k, dt=None: torch.from_numpy(np.ascontiguousarray(a[k][i:i + b])).to(device)  # noqa: E731
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lg = net(t("tiles"), t("globals"), t("slot_kind"), t("mover"))
            outs.append(lg.float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 9), np.float32)


def concat(shards: list[dict], keys: tuple[str, ...] = ("winner", "turn", "kind", "game_key", "mover", "seat", "n_players")) -> dict:
    return {k: np.concatenate([a[k] for a in shards]) for k in keys}


def win_prob(logits: np.ndarray) -> np.ndarray:
    """Slot 0's softmax probability per row."""
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 0] / e.sum(axis=1)


def logit(q: np.ndarray) -> np.ndarray:
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 50) -> tuple[float, float]:
    """(a, b) of the least log loss of y ~ sigmoid(a x + b), by Newton steps with a small pull towards (1, 0)."""
    a, b = 1.0, 0.0
    n = len(x)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * x + b)))
        r = p - y
        w = p * (1 - p)
        g = np.array([(r * x).sum() / n + l2 * (a - 1), r.sum() / n + l2 * b])
        h = np.array([[(w * x * x).sum() / n + l2, (w * x).sum() / n], [(w * x).sum() / n, w.sum() / n + l2]])
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-7:
            break
    return float(a), float(b)


def fit_calibration(logits: np.ndarray, rows: dict, min_rows: int = 500) -> dict[str, list[tuple | None]]:
    """Per kind and stage, the logistic map (a, b) of logit(q) to "slot 0 wins", for the viewer not to move and to
    move (one shared map when either side has fewer than `min_rows` / 4 rows); None with fewer than `min_rows`."""
    y = (rows["winner"] == 0).astype(np.float64)
    x = logit(win_prob(logits))
    mv = rows["mover"].astype(bool)
    st = stage_of(rows["turn"])
    out: dict[str, list[tuple | None]] = {}
    for kind in sorted(set(rows["kind"].tolist())):
        params: list[tuple | None] = []
        for s in range(len(STAGES)):
            sel = (rows["kind"] == kind) & (st == s)
            if sel.sum() < min_rows:
                params.append(None)
                continue
            off, on = sel & ~mv, sel & mv
            if min(off.sum(), on.sum()) < min_rows // 4:
                ab = fit_logistic(x[sel], y[sel])
                params.append(ab + ab)
            else:
                params.append(fit_logistic(x[off], y[off]) + fit_logistic(x[on], y[on]))
        out[kind] = params
    return out


def gate(p: np.ndarray, rows: dict, given: np.ndarray | None = None) -> dict:
    """The gate per kind and stage. `p` [N]: the win chance of slot 0; `given` [N] bool: where a number was given
    (default everywhere). Returns {kind: {"stages": [...], "overall": {...}, "pass": bool}}."""
    y = (rows["winner"] == 0).astype(np.float64)
    st = stage_of(rows["turn"])
    given = np.ones(len(p), bool) if given is None else given
    report = {}
    for kind in sorted(set(rows["kind"].tolist())):
        ks = rows["kind"] == kind
        stages = []
        for s, name in enumerate(STAGE_NAMES):
            sel = ks & (st == s)
            stages.append(stage_report(name, p[sel & given], y[sel & given], rows["game_key"][sel & given], int(sel.sum())))
        overall = stage_report("all", p[ks & given], y[ks & given], rows["game_key"][ks & given], int(ks.sum()))
        checked = [x for x in stages if x["verdict"] not in ("too few games", "no number given")]
        ok = bool(checked) and all(x["verdict"] == "pass" for x in checked)
        report[kind] = {"stages": stages, "overall": overall, "pass": ok,
                        "status": "passes" if ok else "fails" if checked else "gives no number"}
    return report


def stage_report(name: str, p: np.ndarray, y: np.ndarray, games: np.ndarray, rows_all: int) -> dict:
    n = len(p)
    g = len(set(games.tolist())) if n else 0
    r = {"stage": name, "rows": rows_all, "given": n, "games": g}
    if n == 0:
        r["verdict"] = "no number given" if rows_all else "too few games"
        return r
    rel = reliability(p, y)
    base = float(y.mean())
    base_brier = float(((base - y) ** 2).mean())
    worst = 0.0
    for b in rel["buckets"]:
        if b["n"] >= max(200, 0.02 * n):
            worst = max(worst, abs(b["mean_p"] - b["won"]))
    r.update(base_rate=base, brier=rel["brier"], base_brier=base_brier,
             skill=1 - rel["brier"] / base_brier if base_brier > 0 else 0.0, ece=rel["ece"], worst_gap=worst,
             mean_p=float(p.mean()), log_loss=rel["log_loss"], buckets=[
                 {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()} for b in rel["buckets"]])
    if g < MIN_GAMES:
        r["verdict"] = "too few games"
    elif rel["ece"] > MAX_ECE:
        r["verdict"] = "fail: ECE"
    elif rel["brier"] > base_brier:
        r["verdict"] = "fail: worse than the base rate"
    elif worst > MAX_GAP:
        r["verdict"] = "fail: a bucket is off"
    else:
        r["verdict"] = "pass"
    return r


def calibrated(logits: np.ndarray, rows: dict, cal: Calibration | None) -> tuple[np.ndarray, np.ndarray]:
    """(p, given) under a calibration layer: no number where it has no map. `cal` None: the raw win chance."""
    q = win_prob(logits)
    if cal is None:
        return q, np.ones(len(q), bool)
    st = stage_of(rows["turn"])
    ab = np.full((len(q), 2), np.nan)
    for i, (k, s, m) in enumerate(zip(rows["kind"].tolist(), st.tolist(), rows["mover"].tolist())):
        got = cal.get(k, int(s), bool(m))
        if got is not None:
            ab[i] = got
    given = ~np.isnan(ab[:, 0])
    p = np.where(given, 1 / (1 + np.exp(-(np.nan_to_num(ab[:, 0]) * logit(q) + np.nan_to_num(ab[:, 1])))), q)
    return p, given


def format_table(name: str, report: dict) -> str:
    lines = [f"### {name}", ""]
    for kind, r in report.items():
        lines += [f"**{kind}**: {r['status']}", "",
                  "| turns | rows | games | given | base rate | Brier | base Brier | skill | ECE | worst bucket | verdict |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in r["stages"] + [r["overall"]]:
            if "brier" not in s:
                lines.append(f"| {s['stage']} | {s['rows']} | {s['games']} | {s['given']} | | | | | | | {s['verdict']} |")
                continue
            lines.append(f"| {s['stage']} | {s['rows']} | {s['games']} | {s['given']} | {s['base_rate']:.3f} | "
                         f"{s['brier']:.4f} | {s['base_brier']:.4f} | {s['skill']:+.3f} | {s['ece']:.3f} | "
                         f"{s['worst_gap']:.3f} | {s['verdict'] if s['stage'] != 'all' else ''} |")
        lines.append("")
    return "\n".join(lines)


def strip_buckets(report: dict) -> dict:
    """The report without the per-bucket lists (the short JSON)."""
    return {k: {"pass": r["pass"], "status": r["status"], "overall": {a: b for a, b in r["overall"].items() if a != "buckets"},
                "stages": [{a: b for a, b in s.items() if a != "buckets"} for s in r["stages"]]} for k, r in report.items()}


def main(argv: list[str] | None = None) -> int:
    from .train_bc import load_checkpoint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--policy", nargs="*", default=[], help="NAME=CKPT: a policy network's raw value head")
    ap.add_argument("--value", nargs="*", default=[], help="NAME=CKPT: an analysis value checkpoint")
    ap.add_argument("--split", default="test", choices=list(SPLITS))
    ap.add_argument("--keep", nargs="*", help="PREFIX=SHARE: keep that share of the rows of games keyed PREFIX:... (e.g. web=0.3)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out", help="JSON report")
    ap.add_argument("--md", help="Markdown tables")
    a = ap.parse_args(argv)
    device = torch.device(a.device)
    shards = load_splits(a.rows, (a.split,), keep={a.split: parse_keep(a.keep)})[a.split]
    rows = concat(shards)
    print(f"{len(rows['winner'])} rows of {len(set(rows['game_key'].tolist()))} games ({a.split})", flush=True)
    full, tables = {}, []
    for spec in a.policy:
        name, path = spec.split("=", 1)
        net = policy_value_net(load_checkpoint(path).to(device))
        rep = gate(win_prob(predict_logits(net, shards, device, a.batch)), rows)
        full[name] = rep
        tables.append(format_table(f"{name} (raw value head)", rep))
    for spec in a.value:
        name, path = spec.split("=", 1)
        net, cal, _ = load_value(path, device)
        p, given = calibrated(predict_logits(net, shards, device, a.batch), rows, cal)
        rep = gate(p, rows, given)
        full[name] = rep
        tables.append(format_table(f"{name} (calibrated)", rep))
    text = "\n".join(tables)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")
    if a.out:
        Path(a.out).write_text(json.dumps({k: strip_buckets(v) for k, v in full.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
