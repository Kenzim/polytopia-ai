"""Where the time of one self-play rollout goes: inference calls, per-arm engine steps, and the Python in between.

Builds the same Collector as `polyai.rl.train` (learner, BC reference, retail arms, model arms) and plays one
rollout, timing every `choose` call and every `VecEnv.step` per arm.
Usage (from ai/python):
    python ../tools/profile_rollout.py CKPT [--device cuda|cpu] [--games 128] [--steps 8192] [--threads 16]
        [--n-players 2:55,3:25,4:20] [--sizes ...] [--presets ...] [--snapshots 0]
"""
import argparse
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polyai_engine as pe
import torch

from polyai.rl import Collector, Pool, RLConfig
from polyai.rl_pool import EntryInfo, NetPolicy, plan_quotas
from polyai.train_bc import load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--device", default="cuda")
ap.add_argument("--games", type=int, default=128)
ap.add_argument("--steps", type=int, default=8192)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--n-players", default="2:55,3:25,4:20")
ap.add_argument("--sizes", default="")
ap.add_argument("--presets", default="")
ap.add_argument("--snapshots", type=int, default=0, help="frozen copies of the learner in the pool")
ap.add_argument("--max-batch", type=int, default=256)
ap.add_argument("--no-pad-sizes", dest="pad_sizes", action="store_false")
a = ap.parse_args()

torch.set_num_threads(4)
dev = torch.device(a.device)
cfg = RLConfig(games=a.games, rollout_steps=a.steps, threads=a.threads, n_players=a.n_players, sizes=a.sizes,
               presets=a.presets, max_batch=a.max_batch, device=a.device, pad_sizes=a.pad_sizes)
net = load_checkpoint(Path(a.ckpt)).to(dev).eval()
learner = NetPolicy(net, "learner", dev, max_batch=a.max_batch, seed=1)
pool = Pool(cfg, NetPolicy(load_checkpoint(Path(a.ckpt)).to(dev).eval(), "bc", dev, max_batch=a.max_batch, seed=2),
            EntryInfo("bc", "bc"))
for k in range(a.snapshots):
    pool.add_snapshot(f"self:u{k:05d}", NetPolicy(load_checkpoint(Path(a.ckpt)).to(dev).eval(), f"s{k}", dev,
                                                  max_batch=a.max_batch, seed=3 + k), EntryInfo(f"self:u{k:05d}", f"s{k}"))
ex = ThreadPoolExecutor(max_workers=a.threads)
col = Collector(cfg, pe, learner, pool, np.random.default_rng(0), ex)
col.set_quotas(plan_quotas(pool.weights(), cfg.games, col.seat_p, col.shards))

step_t = defaultdict(float)
step_n = defaultdict(int)
step_max_per_tick: list[float] = []
infer_calls = defaultdict(int)
infer_rows = defaultdict(int)
infer_t = defaultdict(float)


def wrap_env(arm):
    orig = arm.env.step

    def step(games, picks, _o=orig, _k=arm.key):
        t0 = time.perf_counter()
        r = _o(games, picks)
        dt = time.perf_counter() - t0
        step_t[_k] += dt
        step_n[_k] += 1
        tick_steps.append(dt)
        return r
    arm.env = type("E", (), {"step": staticmethod(step), "observe": arm.env.observe})()


def wrap_policy(name, p):
    orig = p.choose

    def choose(obs, temperature=None, _o=orig):
        t0 = time.perf_counter()
        r = _o(obs, temperature=temperature)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        infer_t[name] += time.perf_counter() - t0
        infer_calls[name] += 1
        infer_rows[name] += len(r.index)
        return r
    p.choose = choose


for arm in col.arms.values():
    wrap_env(arm)
wrap_policy("learner", learner)
for k, p in pool.nets.items():
    wrap_policy(k, p)

tick_steps: list[float] = []
col.stores, col.results = {}, []
n_ticks = 0
t_start = time.perf_counter()
t_tick_total = 0.0
while col.n_learner < cfg.rollout_steps:
    tick_steps = []
    t0 = time.perf_counter()
    col.tick()
    t_tick_total += time.perf_counter() - t0
    step_max_per_tick.append(max(tick_steps) if tick_steps else 0.0)
    n_ticks += 1
wall = time.perf_counter() - t_start
ex.shutdown()

print(f"rollout: {col.n_learner} learner + {col.n_opp} opponent decisions, {n_ticks} ticks, {len(col.results)} "
      f"games finished, {wall:.1f} s wall ({col.n_learner / wall:.0f} learner dec/s)")
print(f"  infer (all choose calls) {col.t_infer:.1f} s, step phase {col.t_step:.1f} s, "
      f"other {wall - col.t_infer - col.t_step:.1f} s")
print(f"  step phase = slowest arm each tick: sum of per-tick max {sum(step_max_per_tick):.1f} s; "
      f"sum over all arms {sum(step_t.values()):.1f} s (= core-seconds of engine work)")
print("  inference by policy: calls, rows, mean rows/call, seconds")
for k in sorted(infer_calls):
    print(f"    {k:14s} {infer_calls[k]:6d} {infer_rows[k]:8d} {infer_rows[k] / infer_calls[k]:8.1f} {infer_t[k]:7.1f}")
print("  engine step by arm: active games, steps, total s, mean ms")
for k in sorted(step_t, key=lambda k: -step_t[k]):
    print(f"    {str(k):38s} {col.arms[k].n_active:4d} {step_n[k]:6d} {step_t[k]:7.1f} {1e3 * step_t[k] / step_n[k]:7.2f}")
