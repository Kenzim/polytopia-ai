"""Inference throughput of the rollout networks: eager against padded / compiled / CUDA-graph calls (polyai.infer_fast).

Plays a few hundred Might games with the model to get realistic mid-game observations (map sizes, legal-list lengths),
then times `NetPolicy.choose` (collate, forward, sampling, copy back: what a rollout pays) on row subsets of 4 to 256
positions, the variants interleaved round by round so that a shared GPU affects them alike.
Usage (from ai/python):
    python ../tools/bench_infer.py CKPT [--device cuda] [--games 256] [--play-steps 150] [--reps 15] [--gpu-gb 1.5]
        [--variants eager,graphs,compile,compile-graphs]
"""
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import polyai_engine as pe
import torch

from polyai.arena import MODES, mode_kw
from polyai.rl_pool import NetPolicy, merge_padded, obs_len, take
from polyai.train_bc import load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--device", default="cuda")
ap.add_argument("--games", type=int, default=256)
ap.add_argument("--play-steps", type=int, default=150)
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--rows", default="4,8,16,32,64,128,256")
ap.add_argument("--variants", default="eager,graphs,compile-graphs")
ap.add_argument("--gpu-gb", type=float, default=0.0)
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()

torch.set_num_threads(4)
dev = torch.device(a.device)
if dev.type == "cuda":
    dev = torch.device("cuda", torch.cuda.current_device())
    if a.gpu_gb > 0:
        torch.cuda.set_per_process_memory_fraction(a.gpu_gb * 1024**3 / torch.cuda.get_device_properties(dev).total_memory, dev)
VARIANTS = {"eager": {}, "pad": dict(infer_graphs=True), "graphs": dict(infer_graphs=True),
            "compile": dict(infer_compile=True), "compile-graphs": dict(infer_graphs=True, infer_compile=True)}


def make(variant: str, seed: int = 5) -> NetPolicy:
    return NetPolicy(load_checkpoint(Path(a.ckpt)).to(dev).eval(), variant, dev, max_batch=256, seed=seed, **VARIANTS[variant])


# realistic observations: let the model play
env = pe.VecEnv(a.games, 11, "all", split="train", sizes=None, presets=None, turn_limit=150, threads=a.threads, auto_reset=True,
                n_players={2: 55, 3: 25, 4: 20}, seat_difficulties=[1] * 8, **mode_kw(MODES["might"]))
player = make("eager", seed=1)
groups = env.observe()
for _ in range(a.play_steps):
    games, picks = [], []
    for g in groups:
        ch = player.choose(g, temperature=1.0)
        games.append(np.asarray(g["game"], dtype=np.int32))
        picks.append(ch.index.astype(np.int32))
    groups, _ = env.step(np.concatenate(games), np.concatenate(picks))
buckets: dict[int, list] = {}
for g in groups:
    side = max(np.asarray(g["tiles"]).shape[1:3])
    buckets.setdefault(16 if side <= 16 else 20 if side <= 20 else side, []).append(g)
big = max(buckets.values(), key=lambda gs: sum(obs_len(g) for g in gs))
obs_all = merge_padded(big)
n_all = obs_len(obs_all)
lens = np.diff(obs_all["act_off"])
print(f"observations: {n_all} positions of one padded bucket, board {obs_all['tiles'].shape[1:3]}, legal list mean "
      f"{lens.mean():.0f} median {np.median(lens):.0f} max {lens.max()}", flush=True)

rng = np.random.default_rng(0)
variants = a.variants.split(",")
pols = {v: make(v) for v in variants}
rows_list = [r for r in map(int, a.rows.split(",")) if r <= n_all]
subsets = {r: [take(obs_all, np.sort(rng.choice(n_all, r, replace=False))) for _ in range(4)] for r in rows_list}
times: dict[tuple, list] = {(v, r): [] for v in variants for r in rows_list}
for rep in range(a.reps + 2):  # the first two rounds warm up (and capture)
    for r in rows_list:
        for v in variants:
            for obs in subsets[r]:
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                pols[v].choose(obs, temperature=1.0)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                if rep >= 2:
                    times[(v, r)].append(time.perf_counter() - t0)
print("choose() ms per call, median over calls (rows/s in brackets, speed-up over the first variant)")
print(f"{'rows':>5} " + " ".join(f"{v:>28}" for v in variants))
for r in rows_list:
    cells = []
    for v in variants:
        m = statistics.median(times[(v, r)])
        cells.append(f"{1e3 * m:8.2f} ({r / m:8.0f}/s) x{statistics.median(times[(variants[0], r)]) / m:4.2f}")
    print(f"{r:5d} " + " ".join(f"{c:>28}" for c in cells))
if dev.type == "cuda":
    print(f"peak GPU memory allocated {torch.cuda.max_memory_allocated(dev) / 1024**2:.0f} MiB")
for v, p in pols.items():
    if p.fast is not None:
        print(v, p.fast.stats, p.fast.last_error)
