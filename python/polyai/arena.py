"""Model arena: play a trained checkpoint against the retail AI and write results `tools/elo.py` reads.

    nice -n 10 python -m polyai.arena --model /kvm/polytopia-ai-data/runs/NAME \\
        --opponent retail:easy,retail:normal --seeds 50 --out /kvm/polytopia-ai-data/arena/NAME.jsonl

Games are 1v1 Domination from `polyai_engine.VecEnv(split="eval")`, seeds `--seed-start` (1,000,000) upwards. Every seed
is played twice with the model in seat 0 and in seat 1 (same map, same tribe per seat), so seat and tribe advantages
cancel. Setups are drawn from the seed over Tiny/Small/Normal and the five presets, so a few dozen seeds already cover
the suite. Inference is batched on the GPU (bf16 autocast), one forward pass per map size. A game that reaches
`--turn-limit` is a capped game: a draw, `winner` null, whatever the engine's own top-score rule says.

`--model` is a checkpoint file or a run directory (then `ckpt_final.pt`); `PATH:sample=T` plays by sampling from the
policy at temperature T instead of the greedy move. The agent name in the results is `model:<run>` (greedy) or
`model:<run>:sample=T`, next to `retail:<difficulty>`, the same names `polyai-arena` writes.

JSONL (one line per game, the `polyai-arena` format plus a few extra keys): seed, swap (0: first agent in seat 0),
agents (seat 0 first), tribes, setup, winner (seat or null), scores, turns, adjudicated and capped (both: the game
hit the cap), over, commands, illegal, wall_ms (the run's wall time divided by its games, not per-game latency),
think_ms (model inference per seat, amortised), engine, game_version, plus seat_difficulties (the handicaps the
engine used) and model_path. Games already in the file are skipped, so a run can be resumed or extended.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .policy import ModelPolicy, collate, resolve_checkpoint  # noqa: F401  (collate is re-exported)

EVAL_SEED_MIN = 1_000_000
DIFFICULTIES = ["easy", "normal", "hard", "crazy"]  # engine difficulty ids 0 to 3
SIZE_IDS = {"tiny": 1, "small": 2, "normal": 3}
PRESET_IDS = {"dryland": 1, "lakes": 2, "continents": 3, "archipelago": 4, "pangea": 6}
MODEL_SEAT_DIFFICULTY = 1  # the model seat's handicap, as polyai-arena gives a non-retail agent (Normal)
DEFAULT_ARENA_DIR = "/kvm/polytopia-ai-data/arena"


# ---------------------------------------------------------------------------------------------------------------
# Agent names
# ---------------------------------------------------------------------------------------------------------------


def parse_opponent(spec: str) -> int:
    """`retail:easy|normal|hard|crazy` -> engine difficulty id."""
    kind, _, name = spec.partition(":")
    if kind != "retail" or name not in DIFFICULTIES:
        raise ValueError(f"opponent must be retail:{'|'.join(DIFFICULTIES)}, got {spec!r}")
    return DIFFICULTIES.index(name)


def parse_model_spec(spec: str) -> tuple[Path, float | None]:
    """`[model:]PATH[:sample=T]` -> (checkpoint file, temperature or None for greedy). A directory means ckpt_final.pt."""
    if spec.startswith("model:"):
        spec = spec[len("model:"):]
    temperature = None
    head, sep, tail = spec.rpartition(":")
    if sep and tail.startswith("sample="):
        temperature = float(tail[len("sample="):])
        spec = head
    elif sep and tail == "greedy":
        spec = head
    if temperature is not None and temperature <= 0:
        raise ValueError("the sampling temperature must be positive")
    path = resolve_checkpoint(spec)
    return path, temperature


def agent_label(path: Path, temperature: float | None) -> str:
    """`model:<run>` for a run's ckpt_final.pt, `model:<run>/<file stem>` for other checkpoints."""
    run = path.parent.name or "ckpt"
    name = run if path.name == "ckpt_final.pt" else f"{run}/{path.stem}"
    return f"model:{name}" + (f":sample={temperature:g}" if temperature is not None else "")


# ---------------------------------------------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------------------------------------------


class ModelAgent:
    """A checkpoint that picks an index into each legal list: greedy, or sampled at a temperature. The inference is
    `polyai.policy.ModelPolicy`'s, so an arena result measures what the policy interface (and the service) plays."""

    def __init__(self, path: Path, temperature: float | None, device: torch.device, max_batch: int = 128,
                 seed: int = 0):
        self.policy = ModelPolicy(path, device, max_batch, seed)
        self.path, self.temperature, self.max_batch = path, temperature, max_batch
        self.device = device
        self.name = agent_label(path, temperature)
        self.model = self.policy.model
        self.step = self.policy.step

    @property
    def infer_seconds(self) -> float:
        return self.policy.infer_seconds

    @property
    def decisions(self) -> int:
        return self.policy.decisions

    def choose(self, obs: dict) -> np.ndarray:
        return self.policy.choose(obs, temperature=self.temperature).index


# ---------------------------------------------------------------------------------------------------------------
# Playing a pairing
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class Suite:
    """What the games are drawn from. The defaults are the 1v1 Domination main suite."""

    seed_start: int = EVAL_SEED_MIN
    seeds: int = 50
    sizes: list[int] = field(default_factory=lambda: [SIZE_IDS[k] for k in ("tiny", "small", "normal")])
    presets: list[int] = field(default_factory=lambda: list(PRESET_IDS.values()))
    turn_limit: int = 150
    batch_games: int = 64  # games in flight per seat arm
    threads: int = 8  # VecEnv threads of both arms and torch together


def _supports_seat_difficulties(pe) -> bool:
    return "seat_difficulties" in (getattr(pe.VecEnv, "__text_signature__", None) or "")


class Arm:
    """One VecEnv in which the model holds a fixed seat: all seeds of the suite, one seat assignment."""

    def __init__(self, pe, suite: Suite, opp_difficulty: int, model_seat: int, env_threads: int):
        self.pe, self.suite, self.model_seat = pe, suite, model_seat
        self.seed_end = suite.seed_start + suite.seeds
        self.per_seat = _supports_seat_difficulties(pe)
        kw = dict(n_players=[2], sizes=suite.sizes, presets=suite.presets, turn_limit=suite.turn_limit,
                  threads=env_threads, split="eval", auto_reset=True)
        if self.per_seat:
            kw["seat_difficulties"] = [MODEL_SEAT_DIFFICULTY if s == model_seat else opp_difficulty for s in range(2)]
        else:
            kw["difficulty"] = opp_difficulty
        self.env = pe.VecEnv(min(suite.batch_games, suite.seeds), suite.seed_start, [model_seat], **kw)
        self.groups = list(self.env.observe())
        self.finished: dict[int, dict] = {}

    def wanted(self, seed: int) -> bool:
        return self.suite.seed_start <= seed < self.seed_end and seed not in self.finished

    @property
    def done(self) -> bool:
        return len(self.finished) >= self.suite.seeds

    def step(self, agent: ModelAgent) -> None:
        games, choices = [], []
        for g in self.groups:
            keep = np.array([self.wanted(int(s)) for s in g["seed"]])
            if not keep.any():
                continue
            if not keep.all():
                g = _take(g, np.nonzero(keep)[0])
            games.append(np.asarray(g["game"], dtype=np.int32))
            choices.append(agent.choose(g))
        if not games:
            raise RuntimeError("no game is waiting for the model, but not all seeds are finished")
        self.groups, finished = self.env.step(np.concatenate(games), np.concatenate(choices))
        for f in finished:
            if self.wanted(int(f["seed"])):
                self.finished[int(f["seed"])] = f


def _take(g: dict, sel: np.ndarray) -> dict:
    """The positions `sel` of an observation group, with their legal lists re-based."""
    act_off = np.asarray(g["act_off"], dtype=np.int64)
    lens = np.diff(act_off)[sel]
    rows = np.concatenate([np.arange(act_off[i], act_off[i + 1]) for i in sel]) if len(sel) else np.zeros(0, np.int64)
    out = {k: np.asarray(g[k])[sel] for k in ("game", "seat", "n_players", "turn", "seed", "tiles", "globals")}
    out["act_off"] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    out["actions"] = np.asarray(g["actions"])[rows]
    return out


def game_record(f: dict, pe, suite: Suite, swap: int, agents: list[str], setup: dict, model_path: str) -> dict:
    """One finished VecEnv game as a `polyai-arena` JSONL record. Capped games are draws whatever the engine says."""
    capped = bool(f["capped"]) if "capped" in f else bool(f.get("adjudicated", False))
    winners = [int(w) for w in f["winners"]]
    winner = winners[0] if len(winners) == 1 and not capped else None
    return {
        "seed": int(f["seed"]), "swap": swap, "agents": agents, "tribes": [int(t) for t in f["tribes"]],
        "setup": {k: setup[k] for k in ("size", "preset", "mode", "turn_limit")},
        "winner": winner, "scores": [int(x) for x in f["scores"]], "turns": int(f["turns"]),
        "adjudicated": capped, "capped": capped, "over": True, "commands": int(f["commands"]), "illegal": [0, 0],
        "wall_ms": 0.0, "think_ms": [0.0, 0.0], "engine": getattr(pe, "ENGINE_COMMIT", None),
        "game_version": getattr(pe, "GAME_VERSION", None),
        "seat_difficulties": [int(d) for d in f["difficulties"]], "model_path": model_path,
    }


def load_done(path: Path) -> set[tuple[int, int, tuple[str, ...]]]:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["seed"], r["swap"], tuple(r["agents"])))
    return done


def play_pairing(agent: ModelAgent, opponent: str, suite: Suite, done: set | None = None, log=print) -> list[dict]:
    """Play `agent` against `opponent` on all seeds of the suite, both seatings; returns the new JSONL records
    (those not already in `done`)."""
    import polyai_engine as pe

    opp_difficulty = parse_opponent(opponent)
    if suite.seed_start < EVAL_SEED_MIN:
        raise ValueError(f"evaluation seeds start at {EVAL_SEED_MIN}")
    done = done or set()
    torch_threads = torch.get_num_threads()
    env_threads = max(1, (suite.threads - torch_threads) // 2)
    seeds = range(suite.seed_start, suite.seed_start + suite.seeds)
    names = {0: [agent.name, opponent], 1: [opponent, agent.name]}
    # A seat arm whose seeds are all in the file is skipped; a partly finished one is replayed and only the missing
    # games are returned (VecEnv hands out seeds in order, so skipping single seeds would leave slots idle).
    todo = [swap for swap in (0, 1) if any((s, swap, tuple(names[swap])) not in done for s in seeds)]
    if not todo:
        return []
    if not _supports_seat_difficulties(pe):
        log("warning: this polyai_engine has no per-seat difficulties, so the model seat gets the opponent's handicap "
            f"(retail:{DIFFICULTIES[opp_difficulty]}) instead of Normal; results are not comparable with polyai-arena")
    live = [Arm(pe, suite, opp_difficulty, swap, env_threads) for swap in todo]
    t0, last = time.perf_counter(), 0.0
    infer0, dec0 = agent.infer_seconds, agent.decisions
    while not all(a.done for a in live):
        for a in live:
            if not a.done:
                a.step(agent)
        now = time.perf_counter()
        if now - last > 15:
            last = now
            log(f"  {agent.name} vs {opponent}: {sum(len(a.finished) for a in live)}/{suite.seeds * len(live)} games, "
                f"{now - t0:.0f}s")
    wall_ms = (time.perf_counter() - t0) * 1000.0
    n_games = sum(len(a.finished) for a in live)
    think_ms = (agent.infer_seconds - infer0) * 1000.0 / max(n_games, 1)
    records, setups = [], {}
    for a in live:
        swap = a.model_seat
        for seed in sorted(a.finished):
            if (seed, swap, tuple(names[swap])) in done:
                continue
            if seed not in setups:
                setups[seed] = pe.sample_setup(seed, n_players=[2], sizes=suite.sizes, presets=suite.presets,
                                               turn_limit=suite.turn_limit)
            r = game_record(a.finished[seed], pe, suite, swap, names[swap], setups[seed], str(agent.path))
            r["wall_ms"] = wall_ms / max(n_games, 1)
            r["think_ms"] = [think_ms if s == swap else 0.0 for s in range(2)]
            records.append(r)
    return records


def summarise(records: list[dict], model: str, opponent: str) -> dict:
    """W-L-D of `model` against `opponent` over records (draws include capped games)."""
    w = l = d = capped = 0
    turns = []
    for r in records:
        if set(r["agents"]) != {model, opponent}:
            continue
        seat = r["agents"].index(model)
        if r["winner"] is None:
            d += 1
        elif r["winner"] == seat:
            w += 1
        else:
            l += 1
        capped += bool(r.get("capped", r.get("adjudicated")))
        turns.append(r["turns"])
    n = w + l + d
    return {"model": model, "opponent": opponent, "games": n, "wins": w, "losses": l, "draws": d, "capped": capped,
            "score": (w + 0.5 * d) / n if n else None, "mean_turns": float(np.mean(turns)) if turns else None}


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------


def parse_names(text: str | None, table: dict[str, int], what: str) -> list[int] | None:
    if not text:
        return None
    out = []
    for name in text.split(","):
        if name.strip().lower() not in table:
            raise SystemExit(f"unknown {what} {name!r}; choose from {', '.join(table)}")
        out.append(table[name.strip().lower()])
    return out


def main(argv: list[str] | None = None) -> int:
    d = Suite()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True,
                    help="checkpoint file or run directory, optionally PATH:sample=T; repeat for several models")
    ap.add_argument("--opponent", default="retail:easy", help="comma separated, retail:easy|normal|hard|crazy")
    ap.add_argument("--seeds", type=int, default=d.seeds, help="seeds per opponent; each is played with seats swapped")
    ap.add_argument("--seed-start", type=int, default=d.seed_start)
    ap.add_argument("--sizes", default=None, help="comma separated subset of tiny,small,normal (default all)")
    ap.add_argument("--presets", default=None, help="subset of dryland,lakes,continents,archipelago,pangea")
    ap.add_argument("--turn-limit", type=int, default=d.turn_limit, help="compute cap; a game that reaches it is a draw")
    ap.add_argument("--batch-games", type=int, default=d.batch_games, help="games in flight per seat arm")
    ap.add_argument("--max-batch", type=int, default=128, help="most positions per forward pass")
    ap.add_argument("--threads", type=int, default=d.threads, help="VecEnv and torch threads together (at most 8)")
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--gpu-gb", type=float, default=1.2, help="cap on this process's GPU allocator memory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="seed of the sampling generator")
    ap.add_argument("--out", required=True, help="JSONL file; appended to, finished games are skipped")
    a = ap.parse_args(argv)

    threads = max(2, min(a.threads, 8))
    torch.set_num_threads(max(1, min(a.torch_threads, threads - 1)))
    device = torch.device(a.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda":
        from .train_bc import check_gpu_memory

        check_gpu_memory(a.gpu_gb, device)
        torch.cuda.reset_peak_memory_stats(device)
    suite = Suite(seed_start=a.seed_start, seeds=a.seeds,
                  sizes=parse_names(a.sizes, SIZE_IDS, "size") or d.sizes,
                  presets=parse_names(a.presets, PRESET_IDS, "preset") or d.presets,
                  turn_limit=a.turn_limit, batch_games=a.batch_games, threads=threads)
    opponents = [o.strip() for o in a.opponent.split(",") if o.strip()]
    for o in opponents:
        parse_opponent(o)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    summaries = []
    for spec in a.model:
        path, temperature = parse_model_spec(spec)
        agent = ModelAgent(path, temperature, device, a.max_batch, a.seed)
        print(f"{agent.name}: {path} (step {agent.step}), {agent.model.n_params() / 1e6:.2f}M params", flush=True)
        for opp in opponents:
            records = play_pairing(agent, opp, suite, done, log=lambda m: print(m, flush=True))
            with out.open("a") as f:
                for r in records:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
                    done.add((r["seed"], r["swap"], tuple(r["agents"])))
            all_records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
            s = summarise([r for r in all_records if suite.seed_start <= r["seed"] < suite.seed_start + suite.seeds],
                          agent.name, opp)
            summaries.append(s)
            print(f"{opp}: +{len(records)} games", flush=True)
        del agent
    print("\nresult (model against each opponent; a draw is half a win):")
    for s in summaries:
        if s["games"]:
            print(f"  {s['model']} vs {s['opponent']}: W-L-D {s['wins']}-{s['losses']}-{s['draws']} of {s['games']}, "
                  f"score {100 * s['score']:.1f}%, capped {s['capped']}, mean turns {s['mean_turns']:.1f}")
    if device.type == "cuda":
        print(f"GPU peak allocated {torch.cuda.max_memory_allocated(device) / 1024**2:.0f} MiB")
    print(f"results: {out}   report: python tools/elo.py {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
