"""Train the analysis value network on value rows and fit and gate its calibration layer (docs/analysis.md).

usage: python -m polyai.train_value --rows DIR [DIR ...] --init CKPT --out FILE.pt [--steps 20000] [--batch 256]
                                    [--lr 3e-4] [--device cuda] [--gpu-gb 4] [--name NAME] [--report DIR]

`--init` is a policy checkpoint: the value net starts as its trunk and value head. Training uses the `train` split
of `--rows` (rows of undecided games left out), the loss is the cross-entropy of who won (relative slots, slots
without a player masked), with a random board symmetry per batch. Every `--eval-every` steps the loss on part of the
`fit` split is measured and the best weights are kept. `--steps 0` keeps the initial weights: the policy's own value
head plus the calibration layer.

Then the calibration layer: a temperature per (game kind, stage) fitted on the `fit` split (`value_gate.
fit_temperatures`). A (kind, stage) keeps its temperature only when the gate passes it on the `fit` split and on the
`test` split; elsewhere the checkpoint gives no number and records why. `--report DIR` writes the test gate as
`gate.md` and `gate.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .analysis import STAGE_NAMES, Calibration, ValueNet, save_value
from .data import transform_tiles, valid_symmetries
from .value_gate import (calibrated, concat, fit_temperatures, format_table, gate, load_splits, predict_logits,
                         strip_buckets)

KEYS = ("tiles", "globals", "slot_kind", "mover", "winner")


def group_by_shape(shards: list[dict]) -> dict[tuple, dict]:
    groups: dict[tuple, list[dict]] = {}
    for a in shards:
        groups.setdefault(a["tiles"].shape[1:3], []).append(a)
    return {k: {key: np.concatenate([a[key] for a in v]) for key in KEYS} for k, v in groups.items()}


def eval_loss(net: ValueNet, shards: list[dict], device: torch.device) -> float:
    logits = torch.from_numpy(predict_logits(net, shards, device))
    y = torch.from_numpy(np.concatenate([a["winner"] for a in shards]).astype(np.int64))
    return float(F.cross_entropy(logits, y))


def subsample(shards: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
    total = sum(len(a["winner"]) for a in shards)
    if total <= n:
        return shards
    frac = n / total
    out = []
    for a in shards:
        keep = rng.random(len(a["winner"])) < frac
        out.append({k: v[keep] for k, v in a.items()})
    return out


def train(net: ValueNet, train_shards: list[dict], fit_shards: list[dict], a, device: torch.device) -> dict:
    groups = group_by_shape(train_shards)
    shapes = list(groups)
    weights = np.array([len(groups[s]["winner"]) for s in shapes], np.float64)
    weights /= weights.sum()
    rng = np.random.default_rng(a.seed)
    probe = subsample(fit_shards, a.eval_rows, rng)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, a.steps)))))
    best = (eval_loss(net, probe, device), 0, {k: v.detach().clone() for k, v in net.state_dict().items()})
    history = [{"step": 0, "fit_loss": best[0]}]
    print(f"step 0: fit loss {best[0]:.4f}", flush=True)
    t0, run = time.time(), 0.0
    net.train()
    for step in range(1, a.steps + 1):
        shape = shapes[rng.choice(len(shapes), p=weights)]
        g = groups[shape]
        h, w = shape
        b = max(16, a.batch * 196 // (h * w))
        idx = np.sort(rng.integers(0, len(g["winner"]), b))
        tiles = g["tiles"][idx]
        sym = int(rng.choice(valid_symmetries(h, w)))
        if sym:
            tiles = transform_tiles(tiles, sym)
        put = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(device, non_blocking=True)  # noqa: E731
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = net(put(tiles), put(g["globals"][idx]), put(g["slot_kind"][idx]), put(g["mover"][idx]))
        loss = F.cross_entropy(logits.float(), put(g["winner"][idx].astype(np.int64)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * float(loss) if step > 1 else float(loss)
        if step % a.eval_every == 0 or step == a.steps:
            fl = eval_loss(net, probe, device)
            net.train()
            history.append({"step": step, "train_loss": run, "fit_loss": fl})
            print(f"step {step}: train {run:.4f} fit {fl:.4f} ({time.time() - t0:.0f} s)", flush=True)
            if fl < best[0]:
                best = (fl, step, {k: v.detach().clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[2])
    net.eval()
    return {"best_step": best[1], "best_fit_loss": best[0], "history": history}


def gated(logits: np.ndarray, rows: dict, cal: Calibration) -> dict:
    p, given = calibrated(logits, rows, cal)
    return gate(p, rows, given)


def calibrate_and_gate(net: ValueNet, fit_shards: list[dict], test_shards: list[dict], device: torch.device):
    fit_rows, test_rows = concat(fit_shards), concat(test_shards)
    fit_logits = predict_logits(net, fit_shards, device)
    temps = fit_temperatures(fit_logits, fit_rows)
    cal = Calibration(temps)
    fit_rep = gated(fit_logits, fit_rows, cal)
    test_logits = predict_logits(net, test_shards, device)
    test_rep = gated(test_logits, test_rows, cal)
    kept: dict[str, list[float | None]] = {}
    reasons: dict[str, list[str | None]] = {}
    for kind, t in temps.items():
        kt, kr = [], []
        for s, name in enumerate(STAGE_NAMES):
            vf = fit_rep[kind]["stages"][s]["verdict"]
            vt = test_rep.get(kind, {"stages": [{"verdict": "too few games"}] * len(STAGE_NAMES)})["stages"][s]["verdict"]
            if t[s] is not None and vf == "pass" and vt == "pass":
                kt.append(t[s])
                kr.append(None)
            else:
                kt.append(None)
                why = vt if vt != "pass" else vf
                if t[s] is None or why == "too few games":
                    kr.append(f"too few held-out {kind} games at turns {name} to check the win chance")
                else:
                    kr.append(f"the win chance failed the calibration gate for {kind} games at turns {name} ({why})")
        kept[kind], reasons[kind] = kt, kr
    final = Calibration(kept, reasons)
    final_rep = gated(test_logits, test_rows, final)
    raw_rep = gated(test_logits, test_rows, Calibration({k: [1.0] * len(STAGE_NAMES) for k in temps}))
    return final, {"uncalibrated": raw_rep, "temperature": test_rep, "given": final_rep}


def main(argv: list[str] | None = None) -> int:
    from .train_bc import check_gpu_memory, load_checkpoint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--init", required=True, help="policy checkpoint to start from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", help="value id the service reports (default: the out file's stem)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256, help="rows per batch on a 14x14 board (scaled by board area)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-rows", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu-gb", type=float, default=4.0)
    ap.add_argument("--report", help="folder for gate.md and gate.json of the test split")
    a = ap.parse_args(argv)
    device = torch.device(a.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        check_gpu_memory(a.gpu_gb, device)
    torch.manual_seed(a.seed)
    net = ValueNet.from_policy(load_checkpoint(a.init)).to(device)
    splits = load_splits(a.rows, ("train", "fit", "test") if a.steps > 0 else ("fit", "test"))
    fit_shards, test_shards = splits["fit"], splits["test"]
    info = {}
    if a.steps > 0:
        train_shards = splits.pop("train")
        print(f"train {sum(len(x['winner']) for x in train_shards)} rows, fit {sum(len(x['winner']) for x in fit_shards)}, "
              f"test {sum(len(x['winner']) for x in test_shards)}", flush=True)
        info = train(net, train_shards, fit_shards, a, device)
        del train_shards
    net.eval()
    cal, reports = calibrate_and_gate(net, fit_shards, test_shards, device)
    name = a.name or Path(a.out).stem
    meta = {"name": name, "init": str(a.init), "steps": a.steps, "rows": [str(r) for r in a.rows], "train": info,
            "gate": {k: strip_buckets(v) for k, v in reports.items()}}
    save_value(a.out, net, cal, meta)
    text = "\n".join(format_table(f"{name}: {k}", v) for k, v in reports.items())
    print(text)
    if a.report:
        out = Path(a.report)
        out.mkdir(parents=True, exist_ok=True)
        (out / "gate.md").write_text(text + "\n")
        (out / "gate.json").write_text(json.dumps({k: strip_buckets(v) for k, v in reports.items()} | {"train": info}, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
