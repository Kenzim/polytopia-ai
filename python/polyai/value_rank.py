"""Value-head turn-ranking test (docs/research.md recommendation 1): can the value head rank candidate turns?

Per-turn search (polyai.search) scores each candidate turn by the value head at the end of the turn. It added about 110
Elo on bc3 and nothing on rl3. This measures whether the value head orders the turns a search would choose between by
the outcome they really lead to; if it cannot, no search that leans on it (search over sampled maps, search-improved
targets) is worth building until it can.

Method, on diagnostic seeds (from DIAG_SEED_MIN, outside both the training and the arena ranges):
1. Roots. The model (greedy, seat at the Normal handicap) plays 1v1 games against a retail AI; at the first decision of
   our turn on each of SAMPLE_TURNS the game is cloned.
2. Candidates per root: `top` candidates are the network's best `k` first moves, each completed greedily (what the
   search ranks); `sample` candidates are whole turns sampled at temperature 1. A turn ends where the network picks
   EndTurn (as in the search); the value there, from our seat, is the candidate's value V. Identical turns are merged.
3. Realised outcome R: from each candidate's end of turn the game is played to the end `playouts` times, our seat
   sampling at temperature 1 (the temperature the PPO value head was trained under), the other seat the retail AI.
   R is the mean result (win 1, draw 0.5, loss 0).

Reported: within-root Spearman of V and of the prior (the turn's log-probability) against R; how good the turn picked
by V, by the prior, by the search's own score and by the greedy turn are in R; and the split-half reliability of R,
which is the ceiling any ranking can reach with this many playouts.

    nice -n 10 python -m polyai.value_rank --model RUN_OR_CKPT --seeds 32 --workers 16 --opponent retail:crazy \
        --mode might --out /kvm/polytopia-ai-data/diag/value_rank_NAME.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import schema as S
from .policy import ModelPolicy, observe_games
from .search import END_TURN, SearchConfig, SearchPolicy, _terminal_value

DIAG_SEED_MIN = 3_000_000
SAMPLE_TURNS = (2, 4, 7, 11, 16, 22, 30)
TURN_BUDGET = 200  # most commands in one candidate turn
MAX_COMMANDS = 200_000


@dataclass
class Root:
    rid: int
    seed: int
    seat: int
    turn: int
    size: int
    game: object
    value: float = 0.5


@dataclass
class Cand:
    rid: int
    kind: str  # "top" or "sample"
    j: int
    seat: int
    game: object  # advanced through our turn; at the end it is the position before EndTurn
    temperature: float | None
    first_logp: float = 0.0  # log p of the first move (the search's prior)
    logp: float = 0.0  # log p of the whole turn at temperature 1
    line: list = field(default_factory=list)
    left: int = TURN_BUDGET
    value: float | None = None
    end_index: int | None = None  # EndTurn's index at the end position, or None (budget or game over)
    terminal: bool = False
    outcomes: list = field(default_factory=list)


def _by_shape(items):
    groups: dict[tuple, list] = {}
    for it in items:
        groups.setdefault(it.game.shape, []).append(it)
    return list(groups.values())


def _setup(pe, seed: int, seat: int, opp_difficulty: int, mode: int, sizes, presets) -> dict:
    from .arena import MODEL_SEAT_DIFFICULTY, mode_kw

    setup = pe.sample_setup(seed, n_players=[2], sizes=sizes, presets=presets, turn_limit=150, **mode_kw(mode))
    setup["seats"] = [(t, MODEL_SEAT_DIFFICULTY if s == seat else opp_difficulty) for s, (t, _) in enumerate(setup["seats"])]
    return setup


def make_roots(sp: SearchPolicy, seeds: list[int], opp_difficulty: int, mode: int, sizes, presets) -> list[Root]:
    """Play the base games greedily and clone them at the first decision of our turn on each sample turn."""
    import polyai_engine as pe

    lives = []
    for seed in seeds:
        seat = seed % 2
        setup = _setup(pe, seed, seat, opp_difficulty, mode, sizes, presets)
        lives.append({"seed": seed, "seat": seat, "size": setup["size"], "game": pe.Game.from_setup(setup),
                      "last": -1, "n": 0, "done": False})
    roots: list[Root] = []
    while True:
        waiting = []
        for L in lives:
            if L["done"]:
                continue
            g = L["game"]
            while not g.is_over() and g.current_seat() != L["seat"] and L["n"] < MAX_COMMANDS:
                g.play_ai_move()
                L["n"] += 1
            if g.is_over() or L["n"] >= MAX_COMMANDS:
                L["done"] = True
                continue
            t = int(g.turn)
            if t != L["last"]:
                L["last"] = t
                if t in SAMPLE_TURNS:
                    roots.append(Root(len(roots), L["seed"], L["seat"], t, L["size"], g.clone()))
            waiting.append(L)
        if not waiting:
            return roots
        groups: dict[tuple, list] = {}
        for L in waiting:
            groups.setdefault(L["game"].shape, []).append(L)
        for group in groups.values():
            ch = sp.base.choose(observe_games([L["game"] for L in group]), temperature=None)
            for L, i in zip(group, ch.index):
                L["game"].step(int(i))
                L["n"] += 1


def make_candidates(sp: SearchPolicy, roots: list[Root], k: int, n_sample: int) -> list[Cand]:
    """`k` top-first-move candidates completed greedily and `n_sample` sampled turns per root, played to EndTurn."""
    cands: list[Cand] = []
    for group in _by_shape(roots):
        obs = observe_games([r.game for r in group])
        raw = sp._forward(obs, k, None)
        act_off = np.asarray(obs["act_off"])
        for row, r in enumerate(group):
            r.value = float(raw.value[row])
            acts = obs["actions"][act_off[row]:act_off[row + 1]]
            for j in range(k):
                i = int(raw.top_i[row, j])
                if i < 0:
                    continue
                c = Cand(r.rid, "top", j, r.seat, r.game.clone(), None)
                c.first_logp = c.logp = math.log(max(float(raw.top_p1[row, j]), 1e-12))
                action = tuple(int(x) for x in acts[i])
                c.line.append(action)
                if action[0] == END_TURN:
                    c.value, c.end_index = r.value, i
                else:
                    c.game.step(i)
                    c.left -= 1
                cands.append(c)
            for j in range(n_sample):
                cands.append(Cand(r.rid, "sample", j, r.seat, r.game.clone(), 1.0))
    active = [c for c in cands if c.value is None]
    while active:
        live = []
        for c in active:
            g = c.game
            if g.is_over():
                c.value, c.terminal = _terminal_value(g, c.seat), True
            elif g.current_seat() != c.seat:
                c.value = 0.5  # the turn passed without EndTurn; not expected
            else:
                live.append(c)
        groups: dict[tuple, list] = {}
        for c in live:
            groups.setdefault((c.game.shape, c.temperature), []).append(c)
        for (_, temperature), group in groups.items():
            obs = observe_games([c.game for c in group])
            raw = sp._forward(obs, 1, temperature)
            act_off = np.asarray(obs["act_off"])
            for row, c in enumerate(group):
                a = int(raw.index[row])
                action = tuple(int(x) for x in obs["actions"][act_off[row] + a])
                lp = float(raw.logp[row])
                if not c.line:
                    c.first_logp = lp
                c.logp += lp
                c.line.append(action)
                if action[0] == END_TURN or c.left <= 0:
                    c.value = float(raw.value[row])
                    c.end_index = a if action[0] == END_TURN else None
                else:
                    c.game.step(a)
                    c.left -= 1
        active = [c for c in live if c.value is None]
    return cands


def dedupe(cands: list[Cand]) -> list[Cand]:
    seen: dict[tuple, Cand] = {}
    out = []
    for c in cands:
        key = (c.rid, tuple(c.line))
        if key in seen:
            continue
        seen[key] = c
        out.append(c)
    return out


def play_outs(sp: SearchPolicy, cands: list[Cand], playouts: int) -> None:
    """Fill `outcomes` of every candidate: `playouts` games from its end of turn, our seat at temperature 1."""
    jobs = []
    for c in cands:
        if c.terminal:
            c.outcomes = [c.value] * playouts
            continue
        for _ in range(playouts):
            g = c.game.clone()
            if c.end_index is not None:
                g.step(c.end_index)
            jobs.append({"c": c, "game": g, "n": 0, "done": False})
    while True:
        waiting = []
        for J in jobs:
            if J["done"]:
                continue
            g, seat = J["game"], J["c"].seat
            while not g.is_over() and g.current_seat() != seat and J["n"] < MAX_COMMANDS:
                g.play_ai_move()
                J["n"] += 1
            if g.is_over() or J["n"] >= MAX_COMMANDS:
                J["done"] = True
                J["c"].outcomes.append(_terminal_value(g, seat))
            else:
                waiting.append(J)
        if not waiting:
            return
        groups: dict[tuple, list] = {}
        for J in waiting:
            groups.setdefault(J["game"].shape, []).append(J)
        for group in groups.values():
            ch = sp.base.choose(observe_games([J["game"] for J in group]), temperature=1.0)
            for J, i in zip(group, ch.index):
                J["game"].step(int(i))
                J["n"] += 1


def run_seeds(sp: SearchPolicy, seeds: list[int], opp_difficulty: int, mode: int, sizes, presets, k: int, n_sample: int,
              playouts: int) -> list[dict]:
    roots = make_roots(sp, seeds, opp_difficulty, mode, sizes, presets)
    cands = dedupe(make_candidates(sp, roots, k, n_sample))
    play_outs(sp, cands, playouts)
    by_id = {r.rid: r for r in roots}
    out = []
    for c in cands:
        r = by_id[c.rid]
        out.append({"seed": r.seed, "seat": r.seat, "turn": r.turn, "size": r.size, "root_value": r.value,
                    "kind": c.kind, "j": c.j, "value": c.value, "first_logp": c.first_logp, "logp": c.logp,
                    "steps": len(c.line), "terminal": c.terminal, "outcomes": c.outcomes})
    return out


# ---------------------------------------------------------------------------------------------------------------
# Workers and analysis
# ---------------------------------------------------------------------------------------------------------------

_W: dict = {}


def _init(path: str, seed: int) -> None:
    torch.set_num_threads(1)
    _W["sp"] = SearchPolicy(ModelPolicy(path, "cpu", seed=seed), SearchConfig(k=4), seed=seed)


def _work(args) -> list[dict]:
    seeds, opp, mode, sizes, presets, k, n_sample, playouts = args
    return run_seeds(_W["sp"], seeds, opp, mode, sizes, presets, k, n_sample, playouts)


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    ra, rb = _ranks(a), _ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _mean_se(xs) -> tuple[float, float, int]:
    xs = np.asarray([x for x in xs if x is not None], float)
    if len(xs) == 0:
        return float("nan"), float("nan"), 0
    return float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else float("nan"), len(xs)


def _logit(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def analyse(rows: list[dict], w: float = 1.0) -> dict:
    roots: dict[tuple, list[dict]] = {}
    for r in rows:
        roots.setdefault((r["seed"], r["turn"]), []).append(r)
    rho_v, rho_p, rho_half = [], [], []
    picks = {k: [] for k in ("value", "prior", "search", "greedy", "mean", "best")}
    gain_v, gain_s = [], []
    by_phase: dict[str, list] = {"turns 2-7": [], "turns 11-16": [], "turns 22+": []}
    n_multi = 0
    for (seed, turn), cs in roots.items():
        if len(cs) < 2:
            continue
        n_multi += 1
        R = np.array([np.mean(c["outcomes"]) for c in cs])
        V = np.array([c["value"] for c in cs])
        P = np.array([c["logp"] for c in cs])
        rv, rp = spearman(V, R), spearman(P, R)
        rho_v.append(rv)
        rho_p.append(rp)
        phase = "turns 2-7" if turn <= 7 else ("turns 11-16" if turn <= 16 else "turns 22+")
        by_phase[phase].append(rv)
        halves = [(np.mean(c["outcomes"][: len(c["outcomes"]) // 2]), np.mean(c["outcomes"][len(c["outcomes"]) // 2:]))
                  for c in cs if len(c["outcomes"]) >= 2]
        if len(halves) == len(cs):
            rho_half.append(spearman([h[0] for h in halves], [h[1] for h in halves]))
        greedy = [c for c in cs if c["kind"] == "top" and c["j"] == 0]
        tops = [c for c in cs if c["kind"] == "top"]
        r_greedy = float(np.mean(greedy[0]["outcomes"])) if greedy else float(R.mean())
        r_value = float(R[int(np.argmax(V))])
        picks["value"].append(r_value)
        picks["prior"].append(float(R[int(np.argmax(P))]))
        picks["greedy"].append(r_greedy)
        picks["mean"].append(float(R.mean()))
        picks["best"].append(float(R.max()))
        if tops:
            s = [t["first_logp"] + w * _logit(t["value"]) for t in tops]
            r_search = float(np.mean(tops[int(np.argmax(s))]["outcomes"]))
        else:
            r_search = r_greedy
        picks["search"].append(r_search)
        gain_v.append(r_value - r_greedy)
        gain_s.append(r_search - r_greedy)
    half = _mean_se(rho_half)[0]
    ceiling = 2 * half / (1 + half) if half == half and half > -1 else float("nan")
    return {"roots": len(roots), "roots_with_choice": n_multi, "candidates": len(rows),
            "rho_value": _mean_se(rho_v), "rho_prior": _mean_se(rho_p), "rho_split_half": _mean_se(rho_half),
            "reliability_ceiling": ceiling,
            "rho_value_by_phase": {k: _mean_se(v) for k, v in by_phase.items()},
            "pick": {k: _mean_se(v) for k, v in picks.items()},
            "gain_value_over_greedy": _mean_se(gain_v), "gain_search_over_greedy": _mean_se(gain_s)}


def format_report(a: dict) -> str:
    def f(t):
        m, se, n = t
        return f"{m:+.3f} ± {1.96 * se:.3f} (n={n})" if n > 1 else f"{m:+.3f} (n={n})"

    lines = [f"roots {a['roots']}, with a choice {a['roots_with_choice']}, candidates {a['candidates']}",
             f"Spearman(V, R) within roots:      {f(a['rho_value'])}",
             f"Spearman(prior, R) within roots:  {f(a['rho_prior'])}",
             f"split-half Spearman of R:         {f(a['rho_split_half'])}  (Spearman-Brown ceiling "
             f"{a['reliability_ceiling']:.3f})"]
    for k, v in a["rho_value_by_phase"].items():
        lines.append(f"  Spearman(V, R), {k}: {f(v)}")
    lines.append("realised R of the turn picked by:")
    for k, v in a["pick"].items():
        lines.append(f"  {k:<7} {f(v)}")
    lines.append(f"value pick minus greedy:  {f(a['gain_value_over_greedy'])}")
    lines.append(f"search pick minus greedy: {f(a['gain_search_over_greedy'])}")
    return "\n".join(lines)


def main(argv=None) -> None:
    from .arena import MAIN_PRESETS, MAIN_SIZES, MODES, PRESET_IDS, SIZE_IDS, parse_opponent

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--opponent", default="retail:crazy")
    ap.add_argument("--mode", default="might", choices=sorted(MODES))
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--seed-start", type=int, default=DIAG_SEED_MIN)
    ap.add_argument("--k", type=int, default=4, help="top-first-move candidates per root")
    ap.add_argument("--samples", type=int, default=4, help="sampled-turn candidates per root")
    ap.add_argument("--playouts", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds-per-job", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from .policy import resolve_checkpoint

    path = str(resolve_checkpoint(a.model))
    opp = parse_opponent(a.opponent)
    sizes = [SIZE_IDS[s] for s in MAIN_SIZES]
    presets = [PRESET_IDS[p] for p in MAIN_PRESETS]
    seeds = list(range(a.seed_start, a.seed_start + a.seeds))
    jobs = [(seeds[i:i + a.seeds_per_job], opp, MODES[a.mode], sizes, presets, a.k, a.samples, a.playouts)
            for i in range(0, len(seeds), a.seeds_per_job)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(path, 0)) as pool, open(out, "w") as f:
        for done, part in enumerate(pool.imap_unordered(_work, jobs), 1):
            for r in part:
                r["model"] = path
                r["opponent"] = a.opponent
                f.write(json.dumps(r) + "\n")
            f.flush()
            rows += part
            print(f"[{time.time() - t0:7.0f}s] jobs {done}/{len(jobs)}, candidates {len(rows)}", flush=True)
    print(format_report(analyse(rows)))


if __name__ == "__main__":
    main()
