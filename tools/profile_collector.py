"""Learner decisions per second of a rollout collector, with a stand-in policy (no network, no GPU) or a real checkpoint.

With the stand-in policy, which answers in microseconds, this is what the collector itself costs (engine stepping
included): the ceiling a network-bound trainer approaches as inference gets faster. With `--ckpt` the learner, the BC
entry and the snapshots are that network (the end-to-end figure).

Usage (from ai, with PYTHONPATH=python):
    python tools/profile_collector.py [--collector python|rust] [--games 512] [--steps 16384] [--threads 14]
        [--mode might] [--n-players 2:55,3:25,4:20] [--pool-weights ...] [--snapshots 4] [--warmup 1] [--reps 2]
        [--ckpt CKPT --device cuda --gpu-gb 1.5 --infer-graphs --infer-compile] [--max-lead 4] [--min-infer 0] [--seed 0]
        [--features]
Prints decisions per second for each rep.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polyai_engine as pe

from polyai.rl import Collector, Pool, RLConfig
from polyai.rl_pool import EntryInfo, plan_quotas
from polyai.standin import StandInPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--collector", default="python")
ap.add_argument("--games", type=int, default=512)
ap.add_argument("--steps", type=int, default=16384)
ap.add_argument("--threads", type=int, default=14)
ap.add_argument("--mode", default="might")
ap.add_argument("--n-players", default="2:55,3:25,4:20")
ap.add_argument("--pool-weights", default=None, help="e.g. bc=1,retail:easy=0,retail:normal=0,retail:hard=0,retail:crazy=0 for pure self-play")
ap.add_argument("--snapshots", type=int, default=4)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-lead", type=int, default=RLConfig.max_lead)
ap.add_argument("--min-infer", type=int, default=0)
ap.add_argument("--ckpt", default="", help="use this network for every policy instead of the stand-in")
ap.add_argument("--device", default="cuda")
ap.add_argument("--max-batch", type=int, default=256)
ap.add_argument("--gpu-gb", type=float, default=0.0, help="cap this process's GPU memory (0: no cap)")
ap.add_argument("--infer-graphs", action="store_true")
ap.add_argument("--infer-compile", action="store_true")
ap.add_argument("--features", action="store_true", help="collect the engine features too, as the training loop does")
a = ap.parse_args()

extra = {"pool_weights": a.pool_weights} if a.pool_weights else {}
cfg = RLConfig(mode=a.mode, games=a.games, rollout_steps=a.steps, threads=a.threads, n_players=a.n_players,
               max_lead=a.max_lead, min_infer=a.min_infer, max_batch=a.max_batch, device=a.device, features=a.features,
               **extra)
if a.ckpt:
    import torch

    from polyai.rl_pool import NetPolicy
    from polyai.train_bc import load_checkpoint

    torch.set_num_threads(2)
    dev = torch.device(a.device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    if dev.type == "cuda" and a.gpu_gb > 0:
        torch.cuda.set_per_process_memory_fraction(a.gpu_gb * 1024**3 / torch.cuda.get_device_properties(dev).total_memory, dev)
        torch.cuda.reset_peak_memory_stats(dev)
    fast = dict(infer_graphs=a.infer_graphs, infer_compile=a.infer_compile)

    def policy(name: str, seed: int):
        return NetPolicy(load_checkpoint(Path(a.ckpt)).to(dev).eval(), name, dev, max_batch=a.max_batch, seed=seed, **fast)
else:
    def policy(name: str, seed: int):
        return StandInPolicy(name, salt=seed)

learner = policy("learner", 1)
pool = Pool(cfg, policy("bc", 2), EntryInfo("bc", "bc"))
for k in range(a.snapshots):
    pool.add_snapshot(f"self:u{k:05d}", policy(f"s{k}", 3 + k), EntryInfo(f"self:u{k:05d}", f"s{k}"))
ex = ThreadPoolExecutor(max_workers=a.threads)
if a.collector == "rust":
    from polyai.rl_collect import RustCollector as Collector  # noqa: F811
col = Collector(cfg, pe, learner, pool, np.random.default_rng(a.seed), ex)
col.set_quotas(plan_quotas(pool.weights(), cfg.games, col.seat_p, col.shards))
for w in range(a.warmup + a.reps):
    t0 = time.perf_counter()
    col.collect(cfg.rollout_steps)
    wall = time.perf_counter() - t0
    tag = "warmup" if w < a.warmup else "rep"
    print(f"{tag} {w}: {col.n_learner} learner + {col.n_opp} opponent decisions, {len(col.results)} games finished, "
          f"{wall:.1f} s ({col.n_learner / wall:.0f} learner dec/s; infer {col.t_infer:.1f} s, step wait {col.t_step:.1f} s)",
          flush=True)
if a.ckpt and a.device.startswith("cuda"):
    print(f"peak GPU memory allocated {torch.cuda.max_memory_allocated(dev) / 1024**2:.0f} MiB")
ex.shutdown()
