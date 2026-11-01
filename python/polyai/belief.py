"""Auxiliary belief heads: the loss, held-out metrics and the evaluation command (docs/belief.md).

The heads predict, for every tile the mover has not explored, the truth channels of `schema.TRUTH_CHANNELS`. The loss is
a cross entropy per channel, averaged over the hidden tiles of the batch (explored tiles are masked out), then averaged
over the channels. `BeliefMeter` collects confusion matrices and the capital-location ranking on held-out shards.

    python -m polyai.belief --model CKPT --shards DIR_OR_SHARDS... [--batches N] [--out FILE.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import schema as S

CAPITAL = S.TRUTH_NAMES.index("enemy_capital")
TERRAIN = S.TRUTH_NAMES.index("terrain")


def belief_loss(logits: tuple[torch.Tensor, ...], truth: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """logits: per channel [B, V, H, W]; truth int [B, H, W, K]; mask [B, H, W] (1 = hidden tile).
    Returns (mean over channels of the mean cross entropy over hidden tiles, {channel name: its loss})."""
    m = mask.float()
    count = m.sum().clamp(min=1.0)
    per: dict[str, torch.Tensor] = {}
    for k, lg in enumerate(logits):
        ce = F.cross_entropy(lg.float(), truth[..., k].long().clamp(0, S.TRUTH_VOCABS[k] - 1), reduction="none")
        per[S.TRUTH_NAMES[k]] = (ce * m).sum() / count
    return torch.stack(list(per.values())).mean(), per


def scalar_loss(pred: torch.Tensor, truth_g: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """pred, truth_g [B, TRUTH_G]: mean squared error on the labels divided by `schema.TRUTH_G_SCALE`.
    Returns (sum over the scalars of their loss, {name: its loss})."""
    scale = torch.tensor(S.TRUTH_G_SCALE, dtype=torch.float32, device=pred.device)
    err = ((pred.float() - truth_g.float() / scale) ** 2).mean(dim=0)
    per = {name: err[i] for i, name in enumerate(S.TRUTH_G_NAMES)}
    return err.sum(), per


def full_belief_loss(out, truth: torch.Tensor, mask: torch.Tensor, truth_g: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, dict]:
    """The loss of a belief output (`BeliefOutput`): the per-tile cross entropies of `belief_loss` and, when the model
    has the scalar head and `truth_g` is given, the scalar errors, all averaged with equal weight per term."""
    map_loss, per = belief_loss(out.belief, truth, mask)
    if out.scalars is None or truth_g is None:
        return map_loss, per
    s_sum, s_per = scalar_loss(out.scalars, truth_g)
    total = (map_loss * len(per) + s_sum) / (len(per) + len(s_per))
    return total, {**per, **s_per}


def _top_hit_baseline(n: np.ndarray, k: np.ndarray, top: int) -> np.ndarray:
    """P(at least one of `top` tiles drawn uniformly from n candidates, k of them targets, is a target)."""
    miss = np.ones(len(n))
    for i in range(top):
        denom = np.maximum(n - i, 1)
        miss = miss * np.clip((n - k - i) / denom, 0.0, 1.0)
    return np.where(n <= top, 1.0, 1.0 - miss)


class BeliefMeter:
    """Held-out belief metrics, accumulated batch by batch."""

    def __init__(self) -> None:
        self.conf = [np.zeros((v, v), np.int64) for v in S.TRUTH_VOCABS]  # [true, predicted]
        self.ce = np.zeros(S.TRUTH_K)
        self.tiles = 0
        # Where the hidden capitals are: positions with at least one hidden enemy capital on land.
        self.cap_positions = 0
        self.cap_hits = {1: 0.0, 3: 0.0}
        self.cap_base = {1: 0.0, 3: 0.0}

    @torch.no_grad()
    def add(self, logits: tuple[torch.Tensor, ...], truth: torch.Tensor, mask: torch.Tensor) -> None:
        hidden = mask.bool()
        self.tiles += int(hidden.sum())
        for k, lg in enumerate(logits):
            v = S.TRUTH_VOCABS[k]
            y = truth[..., k].long().clamp(0, v - 1)
            ce = F.cross_entropy(lg.float(), y, reduction="none")
            self.ce[k] += float(ce[hidden].sum())
            pred = lg.argmax(dim=1)
            idx = (y[hidden] * v + pred[hidden]).cpu()
            self.conf[k] += torch.bincount(idx, minlength=v * v).reshape(v, v).numpy()
        # Capital location: rank the hidden land tiles by the probability that they hold an enemy capital.
        land = hidden & (truth[..., TERRAIN] != S.WATER_TERRAIN[0]) & (truth[..., TERRAIN] != S.WATER_TERRAIN[1])
        is_cap = (truth[..., CAPITAL] > 0) & land
        p_cap = 1.0 - F.softmax(logits[CAPITAL].float(), dim=1)[:, 0]
        flat_land = land.flatten(1)
        flat_cap = is_cap.flatten(1)
        score = p_cap.flatten(1).masked_fill(~flat_land, -1.0)
        has = flat_cap.any(dim=1)
        if not bool(has.any()):
            return
        n_land = flat_land.sum(dim=1)[has].cpu().numpy().astype(np.float64)
        n_cap = flat_cap.sum(dim=1)[has].cpu().numpy().astype(np.float64)
        top = score.topk(min(3, score.shape[1]), dim=1).indices
        hit = flat_cap.gather(1, top)[has].cpu().numpy()
        self.cap_positions += int(has.sum())
        self.cap_hits[1] += float(hit[:, 0].sum())
        self.cap_hits[3] += float(hit.any(axis=1).sum())
        self.cap_base[1] += float(_top_hit_baseline(n_land, n_cap, 1).sum())
        self.cap_base[3] += float(_top_hit_baseline(n_land, n_cap, 3).sum())

    def result(self) -> dict:
        groups = {}
        for k, name in enumerate(S.TRUTH_NAMES):
            conf = self.conf[k]
            n = int(conf.sum())
            if n == 0:
                continue
            true_counts = conf.sum(axis=1)
            maj = int(true_counts.argmax())
            freq = true_counts / n
            nz = freq[freq > 0]
            entropy = float(-(nz * np.log(nz)).sum())  # cross entropy of a predictor that outputs the class frequencies
            ce = float(self.ce[k] / n)
            pos_true = n - int(true_counts[maj])
            pos_pred = n - int(conf[:, maj].sum())
            keep = np.arange(len(conf)) != maj
            tp = int(conf[keep][:, keep].sum())  # true and predicted both away from the majority class
            prec = tp / pos_pred if pos_pred else float("nan")
            rec = tp / pos_true if pos_true else float("nan")
            groups[name] = {
                "hidden_tiles": n,
                "majority_class": maj,
                "majority_share": float(true_counts[maj] / n),
                "accuracy": float(np.trace(conf) / n),
                "baseline_accuracy": float(true_counts[maj] / n),
                "ce": ce,
                "baseline_ce": entropy,
                "ce_skill": 1.0 - ce / entropy if entropy > 0 else float("nan"),
                "nonmajority_precision": prec,
                "nonmajority_recall": rec,
                "nonmajority_f1": 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec > 0 else float("nan"),
            }
        cap = None
        if self.cap_positions:
            p = self.cap_positions
            cap = {
                "positions_with_hidden_capital": p,
                "top1": self.cap_hits[1] / p, "top1_uniform": self.cap_base[1] / p,
                "top3": self.cap_hits[3] / p, "top3_uniform": self.cap_base[3] / p,
            }
        return {"hidden_tiles": self.tiles, "groups": groups, "capital_location": cap}


def format_table(res: dict) -> str:
    lines = [f"{'channel':14} {'tiles':>10} {'acc':>7} {'base':>7} {'CE':>7} {'CE base':>8} {'skill':>7} {'prec':>6} {'rec':>6}"]
    for name, g in res["groups"].items():
        lines.append(
            f"{name:14} {g['hidden_tiles']:>10} {g['accuracy']:>7.4f} {g['baseline_accuracy']:>7.4f} {g['ce']:>7.4f} "
            f"{g['baseline_ce']:>8.4f} {g['ce_skill']:>7.3f} {g['nonmajority_precision']:>6.3f} {g['nonmajority_recall']:>6.3f}"
        )
    c = res["capital_location"]
    if c:
        lines.append(
            f"capital location over hidden land tiles ({c['positions_with_hidden_capital']} positions): "
            f"top-1 {c['top1']:.3f} (uniform {c['top1_uniform']:.3f}), top-3 {c['top3']:.3f} (uniform {c['top3_uniform']:.3f})"
        )
    return "\n".join(lines)


@torch.no_grad()
def evaluate_belief(model, paths, batch_size: int = 256, max_batches: int | None = None, device="cpu", bf16=False) -> dict:
    from .data import BatchLoader

    if model.belief_heads is None:
        raise SystemExit("the model has no belief heads")
    was = model.training
    model.eval()
    meter = BeliefMeter()
    loader = BatchLoader(paths, batch_size, augment=False, shuffle=False, drop_last=False, device=device, prefetch=2)
    for i, batch in enumerate(loader.iterate(epochs=1)):
        if max_batches is not None and i >= max_batches:
            break
        if "truth" not in batch:
            raise SystemExit("a shard has no truth keys; generate it with polyai-gen --truth")
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=bf16):
            out = model(batch["tiles"], batch["globals"], batch["actions"], batch["action_mask"], belief=True)
        meter.add(out.belief, batch["truth"], batch["truth_mask"])
    model.train(was)
    return meter.result()


def main(argv: list[str] | None = None) -> int:
    from .train_bc import expand_shards, load_checkpoint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint (or run directory) with belief heads")
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--batches", type=int, default=None, help="stop after this many batches (default: all)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write the result as JSON")
    a = ap.parse_args(argv)
    path = Path(a.model)
    if path.is_dir():
        path = path / "ckpt_final.pt"
    model = load_checkpoint(path, belief=True).to(a.device)
    res = evaluate_belief(model, expand_shards(a.shards), a.batch_size, a.batches, a.device, bf16=a.device != "cpu")
    print(format_table(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
