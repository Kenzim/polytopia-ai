"""Head to head: two checkpoints against each other on the 1v1 evaluation suite, JSONL that `tools/elo.py` reads.

    nice -n 10 python -m polyai.h2h --a /kvm/polytopia-ai-data/runs/rl3_b6c96/ckpt_u00240.pt \\
        --b /kvm/polytopia-ai-data/runs/rl1_b6c96_cpu --seeds 200 --device cpu --out /kvm/polytopia-ai-data/arena/h2h.jsonl

Same suite, seeds, seat swap, handicap (both seats Normal) and record format as `polyai.arena`; only the opponent is a
second network instead of the retail AI. Both play greedily unless given `PATH:sample=T`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .arena import (DEFAULT_ARENA_DIR, MODEL_SEAT_DIFFICULTY, MODES, PRESET_IDS, SIZE_IDS, EVAL_SEED_MIN, ModelAgent,
                    Suite, _take, feature_kw, game_record, load_done, mode_kw, parse_model_spec, parse_names, split_tta,
                    summarise)


class PairArm:
    """One VecEnv where `agents[s]` plays seat s, over all seeds of the suite."""

    def __init__(self, pe, suite: Suite, agents: list[ModelAgent], env_threads: int):
        self.suite, self.agents = suite, agents
        self.seed_end = suite.seed_start + suite.seeds
        self.env = pe.VecEnv(min(suite.batch_games, suite.seeds), suite.seed_start, "all", split="eval", n_players=[2],
                             sizes=suite.sizes, presets=suite.presets, turn_limit=suite.turn_limit,
                             threads=env_threads, auto_reset=True,
                             seat_difficulties=[MODEL_SEAT_DIFFICULTY, MODEL_SEAT_DIFFICULTY], **mode_kw(suite.mode),
                             **feature_kw(any(a.features for a in agents)))
        self.groups = list(self.env.observe())
        self.finished: dict[int, dict] = {}

    def wanted(self, seed: int) -> bool:
        return self.suite.seed_start <= seed < self.seed_end and seed not in self.finished

    @property
    def done(self) -> bool:
        return len(self.finished) >= self.suite.seeds

    def step(self) -> None:
        games, choices = [], []
        for g in self.groups:
            seeds, seats = np.asarray(g["seed"]), np.asarray(g["seat"])
            for s, agent in enumerate(self.agents):
                sel = np.nonzero(np.array([self.wanted(int(x)) for x in seeds]) & (seats == s))[0]
                if len(sel) == 0:
                    continue
                sub = _take(g, sel)
                games.append(np.asarray(sub["game"], dtype=np.int32))
                choices.append(agent.choose(sub))
        if not games:
            raise RuntimeError("no game is waiting for a model, but not all seeds are finished")
        self.groups, finished = self.env.step(np.concatenate(games), np.concatenate(choices))
        for f in finished:
            if self.wanted(int(f["seed"])):
                self.finished[int(f["seed"])] = f


def play_h2h(a: ModelAgent, b: ModelAgent, suite: Suite, done: set, log=print) -> list[dict]:
    import polyai_engine as pe

    if suite.seed_start < EVAL_SEED_MIN:
        raise ValueError(f"evaluation seeds start at {EVAL_SEED_MIN}")
    if a.name == b.name:
        raise SystemExit("the two agents have the same name; use different checkpoints or temperatures")
    seeds = range(suite.seed_start, suite.seed_start + suite.seeds)
    seatings = {0: [a, b], 1: [b, a]}
    todo = [sw for sw in (0, 1) if any((s, sw, tuple(x.name for x in seatings[sw])) not in done for s in seeds)]
    env_threads = max(1, (suite.threads - torch.get_num_threads()) // max(len(todo), 1))
    arms = {sw: PairArm(pe, suite, seatings[sw], env_threads) for sw in todo}
    t0, last = time.perf_counter(), 0.0
    while not all(arm.done for arm in arms.values()):
        for arm in arms.values():
            if not arm.done:
                arm.step()
        now = time.perf_counter()
        if now - last > 30:
            last = now
            log(f"  {a.name} vs {b.name}: {sum(len(x.finished) for x in arms.values())}/{suite.seeds * len(arms)} "
                f"games, {now - t0:.0f}s")
    n = max(sum(len(x.finished) for x in arms.values()), 1)
    wall_ms = (time.perf_counter() - t0) * 1000.0 / n
    records = []
    for sw, arm in arms.items():
        names = [x.name for x in seatings[sw]]
        for seed in sorted(arm.finished):
            if (seed, sw, tuple(names)) in done:
                continue
            setup = pe.sample_setup(seed, n_players=[2], sizes=suite.sizes, presets=suite.presets,
                                    turn_limit=suite.turn_limit, **mode_kw(suite.mode))
            r = game_record(arm.finished[seed], pe, suite, sw, names, setup, f"{a.path}|{b.path}")
            r["wall_ms"] = wall_ms
            records.append(r)
    return records


def main(argv: list[str] | None = None) -> int:
    d = Suite()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="checkpoint file or run directory, optionally PATH:sample=T")
    ap.add_argument("--b", required=True, help="the other agent, same form")
    ap.add_argument("--seeds", type=int, default=d.seeds)
    ap.add_argument("--seed-start", type=int, default=d.seed_start)
    ap.add_argument("--sizes", default=None)
    ap.add_argument("--presets", default=None)
    ap.add_argument("--turn-limit", type=int, default=d.turn_limit)
    ap.add_argument("--mode", choices=list(MODES), default="domination")
    ap.add_argument("--batch-games", type=int, default=d.batch_games)
    ap.add_argument("--max-batch", type=int, default=128)
    ap.add_argument("--threads", type=int, default=d.threads, help="VecEnv and torch threads together (at most 8)")
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{DEFAULT_ARENA_DIR}/h2h.jsonl")
    args = ap.parse_args(argv)

    threads = max(2, min(args.threads, 8))
    torch.set_num_threads(max(1, min(args.torch_threads, threads - 1)))
    device = torch.device(args.device)
    suite = Suite(seed_start=args.seed_start, seeds=args.seeds,
                  sizes=parse_names(args.sizes, SIZE_IDS, "size") or d.sizes,
                  presets=parse_names(args.presets, PRESET_IDS, "preset") or d.presets,
                  turn_limit=args.turn_limit, mode=MODES[args.mode], batch_games=args.batch_games, threads=threads)
    agents = []
    for i, spec in enumerate((args.a, args.b)):
        spec, tta = split_tta(spec)
        path, temperature = parse_model_spec(spec)
        agents.append(ModelAgent(path, temperature, device, args.max_batch, args.seed + i, tta=tta))
        print(f"{agents[-1].name}: {path} (step {agents[-1].step})", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    records = play_h2h(agents[0], agents[1], suite, done, log=lambda m: print(m, flush=True))
    with out.open("a") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    all_records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    s = summarise([r for r in all_records if suite.seed_start <= r["seed"] < suite.seed_start + suite.seeds],
                  agents[0].name, agents[1].name)
    print(f"{s['model']} vs {s['opponent']}: W-L-D {s['wins']}-{s['losses']}-{s['draws']} of {s['games']}, "
          f"score {100 * s['score']:.1f}%, capped {s['capped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
