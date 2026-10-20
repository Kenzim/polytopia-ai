"""Behaviour cloning of the retail AI (design.md section 7, step 1).

    nice python -m polyai.train_bc --name bc_b6 --model b6c96 --train DIR_OR_SHARDS --val DIR_OR_SHARDS

Loss = CE(chosen action) + kl_weight * KL(softmax(candidate scores / T) || model, restricted to the candidates)
     + winner CE + Huber loss on the margin (+ a smaller weight on the margin in 5 turns).
`chosen` (the move the AI played, with its noise) is the label; the candidate scores are the AI's noise-free view of
its best moves and only shape the distribution among them. Run directory:
<runs-dir>/<name>/{config.json, metrics.jsonl, ckpt_last.pt, ckpt_final.pt}. `--resume` continues from ckpt_last.pt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from . import schema as S
from .data import BatchLoader
from .model import CONFIGS, ModelOutput, PolyNet, build_model, win_probability

DEFAULT_RUNS_DIR = "/kvm/polytopia-ai-data/runs"
BRIER_TURN_BUCKET = 10


@dataclass
class TrainConfig:
    name: str = "bc"
    model: str = "b6c96"
    train: list[str] | None = None
    val: list[str] | None = None
    batch_size: int = 256
    steps: int = 2000
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    min_lr_frac: float = 0.02
    grad_clip: float = 1.0
    kl_weight: float = 0.5
    kl_temperature: float = 50.0  # candidate scores run to tens of thousands; the median gap of the top two is ~45
    winner_weight: float = 1.0
    margin_weight: float = 0.25
    margin5_weight: float = 0.125
    augment: bool = True
    bf16: bool = True
    eval_every: int = 500
    ckpt_every: int = 500
    eval_batches: int = 50
    log_every: int = 50
    seed: int = 0
    threads: int = 8
    need_gb: float = 2.5
    runs_dir: str = DEFAULT_RUNS_DIR
    device: str = "cuda"
    resume: bool = False


# ---------------------------------------------------------------------------------------------------------------
# Losses and metrics
# ---------------------------------------------------------------------------------------------------------------


def compute_losses(out: ModelOutput, batch: dict[str, torch.Tensor], cfg: TrainConfig) -> dict[str, torch.Tensor]:
    logp = F.log_softmax(out.policy_logits, dim=-1)
    chosen = batch["chosen"]
    ce = -logp.gather(1, chosen[:, None]).squeeze(1).mean()

    cand_idx = batch["cand_idx"]
    # Needs two candidates to say anything: with one the restricted distributions are both a point mass.
    valid = cand_idx >= 0
    has = valid.sum(dim=1) >= 2
    if bool(has.any()):
        # Scores can be +inf (a move the AI sees as decisive): cap them far above any finite score so that they take
        # the whole mass, or share it when there are several.
        score = (batch["cand_score"].float() / cfg.kl_temperature).nan_to_num(posinf=1e4, neginf=-1e4)
        score = score.masked_fill(~valid, -1e9)
        log_t = F.log_softmax(score, dim=-1)
        t = log_t.exp() * valid
        # The model restricted and renormalised to the candidate set, so the KL compares two distributions on it.
        logp_c = logp.gather(1, cand_idx.clamp(min=0)).masked_fill(~valid, -1e9)
        log_q = F.log_softmax(logp_c, dim=-1)
        kl_per = (t * (log_t - log_q)).masked_fill(~valid, 0.0).sum(dim=1)
        kl = (kl_per * has).sum() / has.sum()
    else:
        kl = out.policy_logits.new_zeros(())

    winner = batch["winner"]
    if bool((winner >= 0).any()):
        winner_ce = F.cross_entropy(out.winner_logits.float(), winner, ignore_index=-1)
    else:
        winner_ce = out.winner_logits.new_zeros(())
    # Margins are in thousands of points and run to +/-25, so a squared error would swamp every other term.
    margin = F.smooth_l1_loss(out.margin.float(), batch["margin"].float() / 1000.0)
    margin5 = F.smooth_l1_loss(out.margin5.float(), batch["margin5"].float() / 1000.0)

    total = (
        ce
        + cfg.kl_weight * kl
        + cfg.winner_weight * winner_ce
        + cfg.margin_weight * margin
        + cfg.margin5_weight * margin5
    )
    return {"loss": total, "policy_ce": ce, "policy_kl": kl, "winner_ce": winner_ce, "margin_loss": margin,
            "margin5_loss": margin5}


class Meter:
    """Accumulates held-out metrics: top-1 / top-3 agreement and Brier score of my win probability by turn."""

    def __init__(self) -> None:
        self.n = 0
        self.top1 = 0.0
        self.top3 = 0.0
        self.sums: dict[str, float] = {}
        self.brier: dict[int, list[float]] = {}  # bucket -> [sum of squared error, count]
        self.acc_winner = [0.0, 0]

    @torch.no_grad()
    def add(self, out: ModelOutput, batch: dict[str, torch.Tensor], losses: dict[str, torch.Tensor]) -> None:
        b = batch["chosen"].shape[0]
        k = min(3, out.policy_logits.shape[1])
        top = out.policy_logits.topk(k, dim=1).indices
        hit = top == batch["chosen"][:, None]
        self.top1 += hit[:, :1].any(dim=1).sum().item()
        self.top3 += hit.any(dim=1).sum().item()
        self.n += b
        for name, v in losses.items():
            self.sums[name] = self.sums.get(name, 0.0) + v.item() * b
        p = win_probability(out).float()
        known = batch["winner"] >= 0
        y = (batch["winner"] == 0).float()
        err = ((p - y) ** 2).cpu().numpy()
        bucket = (batch["turn"] // BRIER_TURN_BUCKET).cpu().numpy()
        for e, bk, kn in zip(err, bucket, known.cpu().numpy()):
            if kn:
                cell = self.brier.setdefault(int(bk), [0.0, 0])
                cell[0] += float(e)
                cell[1] += 1

    def result(self) -> dict:
        n = max(self.n, 1)
        tot = sum(c[0] for c in self.brier.values())
        cnt = sum(c[1] for c in self.brier.values())
        return {
            "positions": self.n,
            "top1": self.top1 / n,
            "top3": self.top3 / n,
            **{f"val_{k}": v / n for k, v in self.sums.items()},
            "brier": tot / cnt if cnt else None,
            "brier_by_turn": {
                f"{b * BRIER_TURN_BUCKET}-{(b + 1) * BRIER_TURN_BUCKET - 1}": c[0] / c[1]
                for b, c in sorted(self.brier.items())
            },
        }


@torch.no_grad()
def evaluate(model: PolyNet, loader: BatchLoader, cfg: TrainConfig, device: torch.device) -> dict:
    was_training = model.training
    model.eval()
    meter = Meter()
    for i, batch in enumerate(loader.iterate(epochs=1)):
        if i >= cfg.eval_batches:
            break
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.bf16 and device.type == "cuda"):
            out = model(batch["tiles"], batch["globals"], batch["actions"], batch["action_mask"])
        meter.add(out, batch, compute_losses(out, batch, cfg))
    model.train(was_training)
    return meter.result()


# ---------------------------------------------------------------------------------------------------------------
# Set-up helpers
# ---------------------------------------------------------------------------------------------------------------


def check_gpu_memory(need_gb: float, device: torch.device) -> int:
    """Fail clearly when the GPU has less free memory than the run needs, and cap this process at that amount so it
    can never grow into memory other people are using. Returns the cap in bytes."""
    if device.type != "cuda":
        return 0
    free, total = torch.cuda.mem_get_info(device)
    need = int(need_gb * 1024**3)
    if free < need:
        raise SystemExit(
            f"not enough free GPU memory: {free / 1024**3:.2f} GiB free of {total / 1024**3:.2f} GiB, "
            f"this run needs {need_gb:.2f} GiB. Other processes are using the rest; not starting."
        )
    torch.cuda.set_per_process_memory_fraction(min(1.0, need / total), device)
    return need


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(cfg.steps - cfg.warmup_steps, 1)
    cos = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * cos)


def make_optimizer(model: PolyNet, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    no_decay = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95),
    )


def expand_shards(specs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for s in specs:
        p = Path(s)
        paths.extend(sorted(p.glob("*.npz")) if p.is_dir() else [p])
    if not paths:
        raise SystemExit(f"no shards found in {specs}")
    return paths


def save_checkpoint(path: Path, model: PolyNet, step: int, opt: torch.optim.Optimizer | None = None) -> None:
    state = {"model": model.state_dict(), "model_config": model.cfg.to_dict(), "step": step,
             "encoding_version": S.ENCODING_VERSION}
    if opt is not None:
        state["optimizer"] = opt.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> PolyNet:
    """The model alone, for evaluation or play. `resume_state` loads the optimiser state as well."""
    from .model import ModelConfig

    state = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    return model


def resume_state(path: Path, model: PolyNet, opt: torch.optim.Optimizer, device: torch.device) -> int:
    """Load model and optimiser state from a ckpt_last.pt written by this module and return its step."""
    if not path.exists():
        raise SystemExit(f"cannot resume: {path} does not exist")
    state = torch.load(path, map_location=device, weights_only=False)
    if state["model_config"] != model.cfg.to_dict():
        raise SystemExit(f"cannot resume: {path} was written by a different model configuration")
    if "optimizer" not in state:
        raise SystemExit(f"cannot resume: {path} holds no optimiser state")
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    return int(state["step"])


# ---------------------------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------------------------


def train(cfg: TrainConfig) -> dict:
    torch.set_num_threads(max(1, min(cfg.threads, 24)))
    device = torch.device(cfg.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    cap = check_gpu_memory(cfg.need_gb, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(cfg.seed)

    run_dir = Path(cfg.runs_dir) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg.model).to(device).to(memory_format=torch.channels_last)
    opt = make_optimizer(model, cfg)
    step = 0
    if cfg.resume:
        step = resume_state(run_dir / "ckpt_last.pt", model, opt, device)
        print(f"resumed {cfg.name} from step {step}", flush=True)
    (run_dir / "config.json").write_text(
        json.dumps({"train": asdict(cfg), "model": model.cfg.to_dict(), "params": model.n_params(),
                    "encoding_version": S.ENCODING_VERSION, "started_at_step": step}, indent=2)
    )
    metrics = (run_dir / "metrics.jsonl").open("a", buffering=1)

    def log(rec: dict) -> None:
        metrics.write(json.dumps(rec) + "\n")

    # A resumed run must not replay the batches it has already seen, so the order depends on the step as well.
    train_loader = BatchLoader(expand_shards(cfg.train), cfg.batch_size, augment=cfg.augment,
                               seed=cfg.seed + 1000 * step, device=device)
    val_loader = None
    if cfg.val:
        val_loader = BatchLoader(expand_shards(cfg.val), cfg.batch_size, augment=False, shuffle=False,
                                 drop_last=False, device=device)

    model.train()
    print(f"{cfg.name}: {cfg.model} {model.n_params() / 1e6:.2f}M params, batch {cfg.batch_size}, "
          f"{cfg.steps} steps, cap {cap / 1024**3:.2f} GiB", flush=True)

    sums: dict[str, float] = {}
    count = 0
    t_mark = time.perf_counter()
    last: dict = {}
    sps = 0.0
    steps_mark = step
    for batch in train_loader.iterate(epochs=None):
        if step >= cfg.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.bf16 and device.type == "cuda"):
            out = model(batch["tiles"], batch["globals"], batch["actions"], batch["action_mask"])
        losses = compute_losses(out, batch, cfg)
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        step += 1

        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + v.detach().float().item()
        count += 1

        if step % cfg.log_every == 0 or step == cfg.steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            sps = (step - steps_mark) / (now - t_mark)
            t_mark, steps_mark = now, step
            rec = {"kind": "train", "step": step, "lr": lr_at(step - 1, cfg), "grad_norm": float(gn),
                   "steps_per_sec": sps, **{k: v / count for k, v in sums.items()}}
            if device.type == "cuda":
                rec["max_mem_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
            log(rec)
            print(f"step {step} loss {rec['loss']:.4f} ce {rec['policy_ce']:.4f} win {rec['winner_ce']:.4f} "
                  f"{sps:.2f} steps/s" + (f" peak {rec['max_mem_mb']:.0f} MiB" if "max_mem_mb" in rec else ""),
                  flush=True)
            sums, count = {}, 0
            last = rec

        if val_loader is not None and (step % cfg.eval_every == 0 or step == cfg.steps):
            res = evaluate(model, val_loader, cfg, device)
            log({"kind": "val", "step": step, **res})
            print(f"  val top1 {res['top1']:.3f} top3 {res['top3']:.3f} brier {res['brier']}", flush=True)
            last = {**last, "val": res}
        if step % cfg.ckpt_every == 0 and step < cfg.steps:
            save_checkpoint(run_dir / "ckpt_last.pt", model, step, opt)

    save_checkpoint(run_dir / "ckpt_last.pt", model, step, opt)
    save_checkpoint(run_dir / "ckpt_final.pt", model, step)
    peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    summary = {"name": cfg.name, "model": cfg.model, "params": model.n_params(), "steps": step,
               "steps_per_sec": sps, "peak_mem_mb": peak, "last": last}
    log({"kind": "summary", **{k: v for k, v in summary.items() if k != "last"}})
    metrics.close()
    return summary


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    d = TrainConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=d.name)
    ap.add_argument("--model", default=d.model, choices=sorted(CONFIGS))
    ap.add_argument("--train", nargs="+", required=True, help="shard files or directories of .npz shards")
    ap.add_argument("--val", nargs="+", default=None)
    for f in ("batch_size", "steps", "warmup_steps", "eval_every", "ckpt_every", "eval_batches", "log_every", "seed", "threads"):
        ap.add_argument(f"--{f.replace('_', '-')}", type=int, default=getattr(d, f))
    for f in ("lr", "weight_decay", "min_lr_frac", "grad_clip", "kl_weight", "kl_temperature", "winner_weight",
              "margin_weight", "margin5_weight", "need_gb"):
        ap.add_argument(f"--{f.replace('_', '-')}", type=float, default=getattr(d, f))
    ap.add_argument("--resume", action="store_true", help="continue the run NAME from its ckpt_last.pt")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--runs-dir", default=d.runs_dir)
    ap.add_argument("--device", default=d.device)
    a = ap.parse_args(argv)
    kw = {k: v for k, v in vars(a).items() if k not in ("no_augment", "no_bf16")}
    return TrainConfig(**kw, augment=not a.no_augment, bf16=not a.no_bf16)


def main(argv: list[str] | None = None) -> int:
    summary = train(parse_args(argv))
    print(json.dumps({k: v for k, v in summary.items() if k != "last"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
