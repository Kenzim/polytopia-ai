"""Weight averaging ("model soup") of checkpoints that share an architecture.

    python tools/avg_weights.py --out SOUP.pt A.pt:0.5 B.pt:0.5
    python tools/avg_weights.py --out SOUP.pt A.pt B.pt C.pt        # uniform

Every floating-point tensor in the state dict, BatchNorm running statistics included, is the weighted mean of the
inputs; integer tensors (BatchNorm's num_batches_tracked) are taken from the first input. The weights are
normalised to sum to 1. The inputs must have the same model_config and encoding_version; the output carries them,
plus a `soup` entry listing the inputs and weights, and loads with polyai.train_bc.load_checkpoint like any checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def parse(spec: str) -> tuple[str, float | None]:
    path, sep, w = spec.rpartition(":")
    if sep and path and w.replace(".", "", 1).isdigit():
        return path, float(w)
    return spec, None


def average(specs: list[tuple[str, float | None]]) -> dict:
    weights = [w if w is not None else 1.0 for _, w in specs]
    total = sum(weights)
    if total <= 0:
        raise SystemExit("weights must sum to more than 0")
    weights = [w / total for w in weights]
    ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p, _ in specs]
    first = ckpts[0]
    for (p, _), c in zip(specs[1:], ckpts[1:]):
        if c["model_config"] != first["model_config"] or c.get("encoding_version") != first.get("encoding_version"):
            raise SystemExit(f"{p}: model_config or encoding_version differs from {specs[0][0]}")
        if c["model"].keys() != first["model"].keys():
            raise SystemExit(f"{p}: state dict keys differ from {specs[0][0]}")
    out = {}
    for k, v in first["model"].items():
        if v.is_floating_point():
            out[k] = sum(w * c["model"][k].double() for w, c in zip(weights, ckpts)).to(v.dtype)
        else:
            out[k] = v.clone()
    return {"model": out, "model_config": first["model_config"], "step": 0,
            "encoding_version": first.get("encoding_version"),
            "soup": [{"path": str(Path(p).resolve()), "weight": w} for (p, _), w in zip(specs, weights)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+", help="checkpoint, optionally PATH:WEIGHT")
    a = ap.parse_args()
    ck = average([parse(s) for s in a.inputs])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, a.out)
    h = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()[:12]
    print(f"{a.out}  sha256 {h}  " + ", ".join(f"{Path(s['path']).parent.name}/{Path(s['path']).name}={s['weight']:.3f}"
                                            for s in ck["soup"]))


if __name__ == "__main__":
    main()
