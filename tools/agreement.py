"""How often a model's top move matches the move actually played, on shards (all positions, no augmentation).

    python tools/agreement.py --shards DIR_OR_NPZ... CKPT [CKPT...]

Prints positions, top-1 and top-3 agreement and the policy cross entropy per checkpoint. CPU by default.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch  # noqa: E402

from polyai.data import BatchLoader  # noqa: E402
from polyai.train_bc import TrainConfig, evaluate, expand_shards, load_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args()
    torch.set_num_threads(8)
    device = torch.device(a.device)
    paths = expand_shards(a.shards)
    cfg = TrainConfig(eval_batches=10**9, bf16=False, kl_weight=0.0)
    for ck in a.ckpts:
        model = load_checkpoint(ck).to(device)
        loader = BatchLoader(paths, 256, augment=False, shuffle=False, drop_last=False, device=device)
        r = evaluate(model, loader, cfg, device)
        print(f"{ck}: {r['positions']} positions, top-1 {r['top1']:.3f}, top-3 {r['top3']:.3f}, "
              f"cross entropy {r['val_policy_ce']:.3f}")


if __name__ == "__main__":
    main()
