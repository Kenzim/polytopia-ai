"""Single-thread cost of the pieces of one self-play step: legal list, encode, apply, retail AI move.

Plays games where seat 0 picks uniformly at random and the other seats are the retail AI, timing each call.
Usage: python tools/bench_step.py [n_games] [size]
"""
import sys
import time
from collections import defaultdict

import numpy as np
import polyai_engine as pe

n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
sizes = [int(sys.argv[2])] if len(sys.argv) > 2 else None
rng = np.random.default_rng(0)
t = defaultdict(float)
n = defaultdict(int)
for seed in range(1, n_games + 1):
    g = pe.Game.sample(seed, n_players=[2], sizes=sizes, turn_limit=150)
    while not g.is_over():
        seat = g.current_seat()
        if seat == 0:
            t0 = time.perf_counter(); legal = g.legal_actions(); t1 = time.perf_counter()
            g.encode(); t2 = time.perf_counter()
            g.step(int(rng.integers(len(legal)))); t3 = time.perf_counter()
            t["legal"] += t1 - t0; t["encode"] += t2 - t1; t["apply"] += t3 - t2; n["learner"] += 1
        else:
            t0 = time.perf_counter(); g.play_ai_move(); t["retail_ai"] += time.perf_counter() - t0; n["retail"] += 1
total = sum(t.values())
print(f"{n_games} games, {n['learner']} learner decisions, {n['retail']} retail moves, {total:.1f} s")
for k, v in t.items():
    per = v / (n["retail"] if k == "retail_ai" else n["learner"]) * 1e3
    print(f"  {k:10s} {v:7.2f} s  {100 * v / total:5.1f}%  {per:.3f} ms per call")
print(f"  core-ms per learner decision, all in: {total / n['learner'] * 1e3:.2f}")
