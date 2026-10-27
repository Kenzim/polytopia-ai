"""Per-turn search on top of a `Policy`: look ahead through the rest of our own turn with the value head.

`SearchPolicy` has the `Policy` interface (`choose(obs, temperature=...)`) with one addition: the search needs the live
games, so `choose` takes `games=`, one `polyai_engine.Game` per position of `obs` (in order). See docs/search.md.

At a decision where the network is not sure (its top move has probability below `skip`), the policy's best `k`
moves are each applied to a clone of the game. Every clone is then played on greedily with the network up to the
point where the network would end the turn (or `budget` commands), and the position there is evaluated with the value
head from our seat. A candidate scores `log prior + w * logit(value)`; the best score is played. A line that ends the
game is valued 1, 0 or 0.5 (win, loss, draw) without the network. All rollouts of all positions advance in lockstep,
so every network call is a batch over candidates and games.

The greedy line of the chosen candidate is kept as a plan. The following commands of the turn are taken from it
without a network call, until a step the network was unsure about (and `turns` searches of the turn are not used up),
where the search runs again. A plan is only followed while the live legal list is the one the rollout saw there.

`play_search_pairing` plays a search agent against the retail AI on the arena's 1v1 suite with single `Game`s (the
`VecEnv` does not hand out its games to clone) in a pool of worker processes, and returns the arena's JSONL records.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import schema as S
from .policy import Choices, ModelPolicy, collate, observe_games

END_TURN = S.KIND_ID["EndTurn"]
TOP_SHOWN = 3
VALUE_EPS = 1e-3  # values are clipped to [eps, 1 - eps] before the logit
MAX_COMMANDS = 200_000  # the engine's own safety net against a game that never ends
DEFAULT_WORKERS = 8


@dataclass(frozen=True)
class SearchConfig:
    k: int = 4  # candidates searched per decision; 1 is the plain policy
    w: float = 1.0  # weight of logit(value) against log(prior) in a candidate's score
    skip: float = 0.9  # no search when the network's top move has at least this probability
    budget: int = 200  # most commands a rollout plays after its first move
    turns: int = 4  # most searches per game turn (0 = unlimited)
    plan: bool = True  # follow the chosen line instead of asking the network again

    def __post_init__(self):
        if self.k < 1 or self.budget < 1 or self.turns < 0 or self.w < 0 or not 0 < self.skip <= 1.0001:
            raise ValueError(f"bad search config {self}")

    def tokens(self) -> list[str]:
        """`key=value` for the options that differ from the defaults, after `search=K`."""
        d = SearchConfig()
        out = [f"search={self.k}"]
        for key in ("w", "skip", "budget", "turns"):
            if getattr(self, key) != getattr(d, key):
                out.append(f"{key}={getattr(self, key):g}")
        if self.plan != d.plan:
            out.append(f"plan={int(self.plan)}")
        return out


_SPEC_KEYS = {"search": "k", "w": "w", "skip": "skip", "budget": "budget", "turns": "turns", "plan": "plan"}


def split_search_spec(spec: str) -> tuple[str, SearchConfig | None]:
    """`[model:]PATH[:sample=T][:search=K[:w=..][:skip=..][:budget=..][:turns=..][:plan=0|1]]` -> (spec without the
    search options, config or None when the spec has no `search=`). The options may come in any order after PATH."""
    parts = spec.split(":")
    kept, opts = [], {}
    for p in parts:
        key, eq, val = p.partition("=")
        if eq and key in _SPEC_KEYS:
            opts[_SPEC_KEYS[key]] = val
        else:
            kept.append(p)
    if "k" not in opts:
        if opts:
            raise ValueError(f"{spec!r}: search options without search=K")
        return spec, None
    kw: dict = {"k": int(opts["k"])}
    for key in ("w", "skip"):
        if key in opts:
            kw[key] = float(opts[key])
    for key in ("budget", "turns"):
        if key in opts:
            kw[key] = int(opts[key])
    if "plan" in opts:
        kw["plan"] = opts["plan"] not in ("0", "false", "no")
    return ":".join(kept), SearchConfig(**kw)


# ---------------------------------------------------------------------------------------------------------------
# Plans and rollouts
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class _Step:
    """One greedy command of a rollout, as the network chose it in the position it saw."""

    sig: int  # hash of that position's legal list
    action: tuple  # (kind, arg, src, dst)
    conf: float  # the network's probability of `action`
    value: float  # P(we win) in that position
    top: list  # up to three (action, probability), best first


@dataclass
class _Roll:
    game: object
    seat: int
    left: int
    line: list[_Step] = field(default_factory=list)
    value: float | None = None  # set when the rollout is finished
    fallback: float = 0.5


@dataclass
class _GameState:
    game: object  # strong reference, so that id(game) stays unique while the state is kept
    turn: int = -1
    searches: int = 0
    steps: list[_Step] = field(default_factory=list)
    pos: int = 0


@dataclass
class _Raw:
    """What one network pass says about positions: the pick under the requested temperature, its log-probability and
    the value, plus the best moves by logit with their probabilities at T = 1 and at the requested temperature."""

    index: np.ndarray
    logp: np.ndarray
    value: np.ndarray
    top_i: np.ndarray  # int32[n, k], -1 beyond the legal list
    top_p1: np.ndarray  # float32[n, k]
    top_pt: np.ndarray  # float32[n, k]


def _logit(p: float) -> float:
    p = min(max(float(p), VALUE_EPS), 1.0 - VALUE_EPS)
    return math.log(p / (1.0 - p))


def _subset(obs: dict, sel: np.ndarray) -> dict:
    """The positions `sel` of an observation group with their legal lists re-based."""
    act_off = np.asarray(obs["act_off"], dtype=np.int64)
    lens = np.diff(act_off)[sel]
    rows = np.concatenate([np.arange(act_off[i], act_off[i + 1]) for i in sel]) if len(sel) else np.zeros(0, np.int64)
    out = {k: np.asarray(obs[k])[sel] for k in ("game", "seat", "n_players", "turn", "seed", "tiles", "globals")
           if k in obs}
    out["act_off"] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    out["actions"] = np.asarray(obs["actions"])[rows]
    return out


def _find(actions: np.ndarray, action: tuple) -> int | None:
    hit = np.nonzero((actions == np.asarray(action, dtype=actions.dtype)).all(axis=1))[0]
    return int(hit[0]) if len(hit) else None


def _terminal_value(game, seat: int) -> float:
    winners = game.winners()
    if game.capped() or not winners:
        return 0.5
    return 1.0 if seat in winners else 0.0


class SearchPolicy:
    """A `ModelPolicy` with the per-turn search described in the module docstring.

    `choose(obs, temperature=None, games=[Game, ...])` answers every position of `obs` (one map size). The move is the
    best candidate by score (greedy) or a sample from softmax(score / T). For a searched position `logp` and the
    probabilities in `top` are those of softmax(score / T) over the candidates (T = 1 when greedy) and `value` is the
    value of the chosen line; any other position is answered as `ModelPolicy` would (a followed plan step reports the
    network's probability of the step and a `top` from the rollout). With `k = 1` nothing is ever searched, so the
    answers are exactly the plain policy's. A plan is only followed for a greedy move (`temperature=None`).
    """

    def __init__(self, base: ModelPolicy, config: SearchConfig | None = None, seed: int = 0):
        self.base = base
        self.cfg = config or SearchConfig()
        self.policy_id = base.policy_id + (f"+search{self.cfg.k}" if self.cfg.k > 1 else "")
        self.rng = np.random.default_rng(seed)
        self._state: dict[int, _GameState] = {}
        self.reset_stats()

    # -- statistics -------------------------------------------------------------------------------------------

    def reset_stats(self) -> None:
        self.decisions = 0  # positions answered
        self.searched = 0  # positions where the search ran
        self.followed = 0  # positions answered from a plan, without a network call
        self.root_rows = 0  # network rows for the positions themselves
        self.rollout_rows = 0  # network rows spent in rollouts
        self.infer_seconds = 0.0  # inside the network
        self.seconds = 0.0  # inside `choose`

    @property
    def stats(self) -> dict:
        n = max(self.decisions, 1)
        return {"decisions": self.decisions, "searched": self.searched, "followed": self.followed,
                "root_rows": self.root_rows, "rollout_rows": self.rollout_rows, "infer_seconds": self.infer_seconds,
                "seconds": self.seconds, "s_per_decision": self.seconds / n, "decisions_per_s": n / max(self.seconds, 1e-9)}

    def forget(self, game) -> None:
        """Drop the plan of a finished game."""
        self._state.pop(id(game), None)

    # -- the network ------------------------------------------------------------------------------------------

    @torch.inference_mode()
    def _forward(self, obs: dict, k: int, temperature: float | None) -> _Raw:
        base = self.base
        act_off = np.asarray(obs["act_off"], dtype=np.int64)
        n = len(act_off) - 1
        lens = np.diff(act_off)
        if (lens <= 0).any():
            raise ValueError("a position has an empty legal list")
        index = np.empty(n, np.int32)
        logp = np.empty(n, np.float32)
        value = np.empty(n, np.float32)
        top_i = np.full((n, k), -1, np.int32)
        top_p1 = np.zeros((n, k), np.float32)
        top_pt = np.zeros((n, k), np.float32)
        t0 = time.perf_counter()
        for s in range(0, n, base.max_batch):
            sel = np.arange(s, min(n, s + base.max_batch))
            tiles, globals_, actions, mask = collate(obs, sel, base.device)
            with torch.autocast(base.device.type, dtype=torch.bfloat16, enabled=base.device.type == "cuda"):
                out = base.model(tiles, globals_, actions, mask)
            logits = out.policy_logits.float()
            lp1 = torch.log_softmax(logits, dim=1)
            if temperature is None:
                pick, lpt = logits.argmax(dim=1), lp1
            else:
                lpt = torch.log_softmax(logits / temperature, dim=1)
                pick = torch.multinomial(lpt.exp(), 1, generator=base.generator)[:, 0]
            kk = min(k, logits.shape[1])
            ti = lp1.topk(kk, dim=1).indices
            index[sel] = pick.cpu().numpy()
            logp[sel] = lpt.gather(1, pick[:, None])[:, 0].cpu().numpy()
            value[sel] = torch.softmax(out.winner_logits.float(), dim=-1)[:, 0].cpu().numpy()
            ti_np = ti.cpu().numpy()
            top_i[sel, :kk] = ti_np
            top_p1[sel, :kk] = lp1.gather(1, ti).exp().cpu().numpy()
            top_pt[sel, :kk] = lpt.gather(1, ti).exp().cpu().numpy()
        bad = top_i >= lens[:, None]
        top_i[bad], top_p1[bad], top_pt[bad] = -1, 0.0, 0.0
        self.infer_seconds += time.perf_counter() - t0
        return _Raw(index, logp, value, top_i, top_p1, top_pt)

    # -- rollouts ---------------------------------------------------------------------------------------------

    def _run_rollouts(self, rolls: list[_Roll]) -> None:
        """Play every unfinished rollout greedily to the point where the network ends the turn (or its budget runs
        out), all of them advancing together so each network call is one batch."""
        active = [r for r in rolls if r.value is None]
        while active:
            live = []
            for r in active:
                if r.game.is_over():
                    r.value = _terminal_value(r.game, r.seat)
                elif r.game.current_seat() != r.seat:
                    r.value = r.fallback  # the turn passed without an EndTurn step; not expected
                else:
                    live.append(r)
            if not live:
                break
            obs = observe_games([r.game for r in live])
            raw = self._forward(obs, TOP_SHOWN, None)
            self.rollout_rows += len(live)
            act_off = np.asarray(obs["act_off"])
            for j, r in enumerate(live):
                acts = obs["actions"][act_off[j]:act_off[j + 1]]
                a = int(raw.index[j])
                top = [(tuple(int(x) for x in acts[i]), float(p)) for i, p in zip(raw.top_i[j], raw.top_pt[j]) if i >= 0]
                step = _Step(hash(acts.tobytes()), tuple(int(x) for x in acts[a]), float(math.exp(raw.logp[j])),
                             float(raw.value[j]), top)
                r.line.append(step)
                if step.action[0] == END_TURN or r.left <= 0:
                    r.value = step.value
                else:
                    r.game.step(a)
                    r.left -= 1
            active = [r for r in live if r.value is None]

    # -- the policy -------------------------------------------------------------------------------------------

    def _game_state(self, game) -> _GameState:
        st = self._state.get(id(game))
        if st is None:
            st = self._state[id(game)] = _GameState(game)
        turn = int(game.turn)
        if st.turn != turn:
            st.turn, st.searches, st.steps, st.pos = turn, 0, [], 0
        return st

    def _may_search(self, st: _GameState, n_legal: int) -> bool:
        return self.cfg.k > 1 and n_legal > 1 and (self.cfg.turns == 0 or st.searches < self.cfg.turns)

    def choose(self, obs: dict, *, temperature: float | None = None, games=None) -> Choices:
        if temperature is not None and not temperature > 0:
            raise ValueError("the sampling temperature must be positive (None is the greedy move)")
        act_off = np.asarray(obs["act_off"], dtype=np.int64)
        n = len(act_off) - 1
        lens = np.diff(act_off)
        if games is None or len(games) != n:
            raise ValueError("SearchPolicy.choose needs games=: one polyai_engine.Game per position of obs")
        cfg, t_start = self.cfg, time.perf_counter()
        index = np.zeros(n, np.int32)
        logp = np.zeros(n, np.float32)
        value = np.zeros(n, np.float32)
        top: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        states = [self._game_state(g) for g in games]
        rows = [np.asarray(obs["actions"])[act_off[i]:act_off[i + 1]] for i in range(n)]

        roots = []
        for i in range(n):
            st, step = states[i], None
            if cfg.plan and temperature is None and st.pos < len(st.steps):
                step = st.steps[st.pos]
                if step.sig != hash(rows[i].tobytes()) or _find(rows[i], step.action) is None:
                    st.steps, st.pos, step = [], 0, None  # the game is not where the plan left it
            if step is not None and not (step.conf < cfg.skip and self._may_search(st, int(lens[i]))):
                a = _find(rows[i], step.action)
                st.pos += 1
                index[i], logp[i], value[i] = a, math.log(max(step.conf, 1e-12)), step.value
                top[i] = [(j, p) for act, p in step.top if (j := _find(rows[i], act)) is not None]
                self.followed += 1
            else:
                roots.append(i)

        searching: list[tuple[int, list[int], list[float], _Raw, int]] = []
        if roots:
            sel = np.asarray(roots)
            sub = obs if len(roots) == n else _subset(obs, sel)
            raw = self._forward(sub, max(cfg.k, TOP_SHOWN), temperature)
            self.root_rows += len(roots)
            for r, i in enumerate(roots):
                if self._may_search(states[i], int(lens[i])) and raw.top_p1[r, 0] < cfg.skip:
                    searching.append((i, [], [], raw, r))
                    continue
                index[i], logp[i], value[i] = raw.index[r], raw.logp[r], raw.value[r]
                top[i] = [(int(j), float(p)) for j, p in zip(raw.top_i[r, :TOP_SHOWN], raw.top_pt[r, :TOP_SHOWN]) if j >= 0]
                states[i].steps, states[i].pos = [], 0
        if searching:
            self._search(searching, games, states, rows, index, logp, value, top, temperature)
        self.decisions += n
        self.seconds += time.perf_counter() - t_start
        return Choices(index, logp, value, top)

    def _search(self, searching, games, states, rows, index, logp, value, top, temperature) -> None:
        cfg = self.cfg
        rolls_of: dict[int, list[tuple[int, float, _Roll]]] = {}
        all_rolls: list[_Roll] = []
        for i, _, _, raw, r in searching:
            game, seat = games[i], games[i].current_seat()
            if seat is None:
                raise ValueError("a position to search must be on a seat's turn, not Nature's")
            root_value = float(raw.value[r])
            lst = []
            for a, p in zip(raw.top_i[r, :cfg.k], raw.top_p1[r, :cfg.k]):
                if a < 0:
                    continue
                act = tuple(int(x) for x in rows[i][a])
                roll = _Roll(None, seat, cfg.budget, fallback=root_value)
                if act[0] == END_TURN:
                    roll.value = root_value  # ending the turn now: the position as it stands is the one to value
                else:
                    roll.game = game.clone()
                    roll.game.step(int(a))
                lst.append((int(a), float(p), roll))
                all_rolls.append(roll)
            rolls_of[i] = lst
        self._run_rollouts(all_rolls)
        for i, _, _, _, _ in searching:
            cands = rolls_of[i]
            score = np.array([math.log(max(p, 1e-12)) + cfg.w * _logit(roll.value) for _, p, roll in cands])
            if temperature is None:
                pick, scale = int(np.argmax(score)), 1.0
            else:
                scale = temperature
            z = score / scale
            lp = z - (z.max() + math.log(np.exp(z - z.max()).sum()))
            if temperature is not None:
                pick = int(self.rng.choice(len(cands), p=np.exp(lp) / np.exp(lp).sum()))
            a, _, roll = cands[pick]
            index[i], logp[i], value[i] = a, lp[pick], roll.value
            order = np.argsort(-score)[:TOP_SHOWN]
            top[i] = [(cands[j][0], float(math.exp(lp[j]))) for j in order]
            st = states[i]
            st.searches += 1
            # The line holds the commands after the chosen one (none when that one ends the turn).
            st.steps, st.pos = (roll.line if cfg.plan and temperature is None else []), 0
            self.searched += 1


def load_search_policy(path: str | Path, config: SearchConfig, device: str = "cpu", seed: int = 0) -> SearchPolicy:
    return SearchPolicy(ModelPolicy(path, device, seed=seed), config, seed=seed)


# ---------------------------------------------------------------------------------------------------------------
# The arena with single games
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class _Live:
    seed: int
    swap: int  # the seat the model plays
    game: object
    setup: dict
    commands: int = 0
    done: bool = False


def _search_setup(pe, suite, seed: int, model_seat: int, opp_difficulty: int) -> dict:
    """The setup `Arm` plays: the suite's draw for the seed, the model seat at the Normal handicap."""
    from .arena import MODEL_SEAT_DIFFICULTY, mode_kw

    setup = pe.sample_setup(seed, n_players=[2], sizes=suite.sizes, presets=suite.presets, turn_limit=suite.turn_limit,
                            **mode_kw(getattr(suite, "mode", 2)))
    setup["seats"] = [(t, MODEL_SEAT_DIFFICULTY if s == model_seat else opp_difficulty)
                      for s, (t, _) in enumerate(setup["seats"])]
    return setup


def play_games(policy: SearchPolicy, jobs: list[tuple[int, int]], suite, opp_difficulty: int,
               temperature: float | None = None) -> list[dict]:
    """Play `(seed, model seat)` jobs together, one game each, the model seat through `policy` and the other seat by
    the retail AI. Positions that wait for the model are answered in one batch per map size. Returns one finished-game
    dict per job (the keys `arena.game_record` reads, plus `setup`, `swap`, `wall_s`)."""
    import polyai_engine as pe

    live = [_Live(seed, swap, None, None) for seed, swap in jobs]
    for L in live:
        L.setup = _search_setup(pe, suite, L.seed, L.swap, opp_difficulty)
        L.game = pe.Game.from_setup(L.setup)
    t0 = time.perf_counter()
    finished: list[dict] = []
    while True:
        waiting = []
        for L in live:
            if L.done:
                continue
            g = L.game
            while not g.is_over() and L.commands < MAX_COMMANDS and g.current_seat() != L.swap:
                g.play_ai_move()  # the retail AI at its handicap, or Nature
                L.commands += 1
            if g.is_over() or L.commands >= MAX_COMMANDS:
                L.done = True
                policy.forget(g)
                finished.append({
                    "seed": L.seed, "swap": L.swap, "setup": L.setup, "capped": g.capped(),
                    "winners": g.winners(), "tribes": [t for t, _ in L.setup["seats"]], "scores": g.scores(),
                    "turns": g.turn, "commands": L.commands, "difficulties": [d for _, d in L.setup["seats"]],
                    "wall_s": time.perf_counter() - t0})
            else:
                waiting.append(L)
        if not waiting:
            return finished
        by_shape: dict[tuple, list[_Live]] = {}
        for L in waiting:
            by_shape.setdefault(L.game.shape, []).append(L)
        for group in by_shape.values():
            games = [L.game for L in group]
            choices = policy.choose(observe_games(games), temperature=temperature, games=games)
            for L, index in zip(group, choices.index):
                L.game.step(int(index))
                L.commands += 1


_WORKER: dict = {}


def _init_worker(path: str, cfg: SearchConfig, seed: int, max_batch: int) -> None:
    torch.set_num_threads(1)
    base = ModelPolicy(path, "cpu", max_batch, seed)
    _WORKER["policy"] = SearchPolicy(base, cfg, seed)


def _work(args) -> tuple[list[dict], dict]:
    jobs, suite, opp_difficulty, temperature = args
    policy: SearchPolicy = _WORKER["policy"]
    policy.reset_stats()
    t0 = time.perf_counter()
    finished = play_games(policy, jobs, suite, opp_difficulty, temperature)
    stats = policy.stats
    stats["wall_s"] = time.perf_counter() - t0
    stats["games"] = len(jobs)
    return finished, stats


def _add_stats(total: dict, s: dict) -> None:
    for key, v in s.items():
        if key in ("decisions", "searched", "followed", "root_rows", "rollout_rows", "infer_seconds", "seconds",
                   "wall_s", "games"):
            total[key] = total.get(key, 0) + v


def play_search_pairing(path: Path, cfg: SearchConfig, temperature: float | None, opponent: str, suite,
                        done: set | None = None, log=print, workers: int = DEFAULT_WORKERS, inflight: int = 4,
                        seed: int = 0, max_batch: int = 128, name: str | None = None) -> tuple[list[dict], dict]:
    """The search agent against `opponent` on all seeds of the suite, both seatings, like `arena.play_pairing`.
    Returns (new JSONL records, summed policy statistics). `workers` processes of one torch thread each play chunks of
    `inflight` games; `workers <= 1` plays in this process with the current torch thread count."""
    import polyai_engine as pe

    from .arena import EVAL_SEED_MIN, game_record, parse_opponent

    opp_difficulty = parse_opponent(opponent)
    if suite.seed_start < EVAL_SEED_MIN:
        raise ValueError(f"evaluation seeds start at {EVAL_SEED_MIN}")
    done = done or set()
    name = name or search_agent_name(path, cfg, temperature)
    names = {0: [name, opponent], 1: [opponent, name]}
    seeds = range(suite.seed_start, suite.seed_start + suite.seeds)
    jobs = [(s, swap) for swap in (0, 1) for s in seeds if (s, swap, tuple(names[swap])) not in done]
    if not jobs:
        return [], {}
    rng = np.random.default_rng(seed)
    rng.shuffle(jobs)  # long games are not all in one chunk
    inflight = max(1, min(inflight, -(-len(jobs) // max(workers, 1))))  # keep every worker busy
    chunks = [jobs[i:i + inflight] for i in range(0, len(jobs), inflight)]
    args = [(c, suite, opp_difficulty, temperature) for c in chunks]
    total: dict = {}
    results: list[dict] = []
    t0, last = time.perf_counter(), 0.0

    def progress() -> None:
        nonlocal last
        now = time.perf_counter()
        if now - last > 15:
            last = now
            log(f"  {name} vs {opponent}: {len(results)}/{len(jobs)} games, {now - t0:.0f}s")

    if workers <= 1:
        _WORKER["policy"] = SearchPolicy(ModelPolicy(path, "cpu", max_batch, seed), cfg, seed)
        for a in args:
            fin, st = _work(a)
            results += fin
            _add_stats(total, st)
            progress()
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(workers, len(chunks)), _init_worker, (str(path), cfg, seed, max_batch)) as pool:
            for fin, st in pool.imap_unordered(_work, args):
                results += fin
                _add_stats(total, st)
                progress()
    wall = time.perf_counter() - t0
    total["elapsed_s"] = wall
    records = []
    for f in sorted(results, key=lambda f: (f["swap"], f["seed"])):
        r = game_record(f, pe, suite, f["swap"], names[f["swap"]], f["setup"], str(path))
        r["wall_ms"] = wall * 1000.0 / max(len(jobs), 1)
        r["think_ms"] = [total["seconds"] * 1000.0 / max(len(jobs), 1) if s == f["swap"] else 0.0 for s in range(2)]
        records.append(r)
    return records, total


def search_agent_name(path: Path, cfg: SearchConfig, temperature: float | None) -> str:
    """`model:<run>[:sample=T]:search=K[:option=value...]`, the agent name in the results."""
    from .arena import agent_label

    return ":".join([agent_label(path, temperature)] + cfg.tokens())


def describe_stats(total: dict) -> str:
    if not total:
        return "nothing played"
    n = max(total["decisions"], 1)
    return (f"{total['games']} games, {total['decisions']} decisions: {total['seconds'] / n * 1000:.1f} ms each "
            f"({n / max(total['seconds'], 1e-9):.0f}/s), searched {total['searched']} "
            f"({100 * total['searched'] / n:.0f}%), plan steps {total['followed']}, "
            f"network rows {total['root_rows']} root + {total['rollout_rows']} rollout; "
            f"{total['seconds'] / max(total['games'], 1):.1f} s of policy per game, wall {total['elapsed_s']:.0f}s")


def run_arena_spec(path: Path, cfg: SearchConfig, temperature: float | None, opponents: list[str], suite, done: set,
                   out: Path, log=print, workers: int = DEFAULT_WORKERS, inflight: int = 4, seed: int = 0,
                   max_batch: int = 128) -> list[dict]:
    """What `arena.main` does for a `PATH:search=K` spec: play each opponent, append the records to `out` and return
    the `arena.summarise` dict per opponent."""
    import json

    from .arena import summarise

    summaries = []
    name = search_agent_name(path, cfg, temperature)
    for opp in opponents:
        records, total = play_search_pairing(path, cfg, temperature, opp, suite, done, log, workers, inflight, seed,
                                             max_batch)
        with out.open("a") as f:
            for r in records:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
                done.add((r["seed"], r["swap"], tuple(r["agents"])))
        log(f"{opp}: +{len(records)} games; {describe_stats(total)}")
        all_records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        summaries.append(summarise([r for r in all_records
                                    if suite.seed_start <= r["seed"] < suite.seed_start + suite.seeds], name, opp))
    return summaries
