"""Helpers of the PPO self-play trainer (`polyai.rl`): the batched network policy, observation batching, the opponent
pool (running win rates, sampling weights) and the plan that turns pool weights into games per VecEnv arm.

Nothing here touches the engine or the optimiser, so it is all testable with a tiny model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .model import PolyNet
from .policy import ModelPolicy

RETAIL_NAMES = ["easy", "normal", "hard", "crazy"]  # engine difficulty ids 0 to 3
LEARNER_DIFFICULTY = 1  # the learner's own handicap: Normal
MAX_SEATS = 8  # the largest player count the encoding holds; seats no game reaches get probability 0 and no arm


# ---------------------------------------------------------------------------------------------------------------
# Observation batches (the dicts VecEnv returns, one map size each)
# ---------------------------------------------------------------------------------------------------------------

OBS_ROW_KEYS = ("game", "seat", "n_players", "turn", "seed", "tiles", "globals")


def obs_len(g: dict) -> int:
    return len(g["act_off"]) - 1


def take(g: dict, sel: np.ndarray) -> dict:
    """The positions `sel` of an observation group as a new group, with the legal lists re-based."""
    act_off = np.asarray(g["act_off"], dtype=np.int64)
    lens = np.diff(act_off)[sel]
    if len(sel):
        rows = np.concatenate([np.arange(act_off[i], act_off[i + 1]) for i in sel])
    else:
        rows = np.zeros(0, np.int64)
    out = {k: np.asarray(g[k])[sel] for k in OBS_ROW_KEYS if k in g}
    out["act_off"] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    out["actions"] = np.asarray(g["actions"])[rows]
    return out


def merge(groups: list[dict]) -> dict:
    """Concatenate observation groups of one map size into one."""
    if len(groups) == 1:
        return groups[0]
    lens = np.concatenate([np.diff(np.asarray(g["act_off"], dtype=np.int64)) for g in groups])
    out = {k: np.concatenate([np.asarray(g[k]) for g in groups]) for k in OBS_ROW_KEYS if k in groups[0]}
    out["act_off"] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    out["actions"] = np.concatenate([np.asarray(g["actions"]) for g in groups])
    return out


def pad_bucket(hw: tuple[int, int]) -> int:
    """Map sizes that share one padded inference batch: up to Normal (16), Large and Huge (20), then each alone.
    Padding Tiny to Normal costs little; padding everything to Massive (30) would cost several times the convolutions."""
    side = max(hw)
    return 16 if side <= 16 else 20 if side <= 20 else side


def merge_padded(groups: list[dict]) -> dict:
    """Concatenate observation groups of different map sizes into one, each board in the top-left corner of the
    largest: tiles are zero-padded, the tile indices of the actions are re-based to the padded width, and a
    `board_mask` marks the real cells (the network then sees each board as on its own, model.py). Groups of a single
    size are merged without padding."""
    shapes = {tuple(np.asarray(g["tiles"]).shape[1:3]) for g in groups}
    if len(shapes) == 1:
        return merge(groups)
    hp = max(h for h, _ in shapes)
    wp = max(w for _, w in shapes)
    tiles, masks, actions = [], [], []
    for g in groups:
        t = np.asarray(g["tiles"])
        n, h, w = t.shape[:3]
        pt = np.zeros((n, hp, wp, t.shape[3]), t.dtype)
        pt[:, :h, :w] = t
        tiles.append(pt)
        m = np.zeros((n, hp, wp), bool)
        m[:, :h, :w] = True
        masks.append(m)
        a = np.array(g["actions"], dtype=np.int16, copy=True)
        for c in (2, 3):
            idx = a[:, c].astype(np.int32)
            real = idx >= 0
            a[real, c] = (idx[real] // w * wp + idx[real] % w).astype(np.int16)
        actions.append(a)
    lens = np.concatenate([np.diff(np.asarray(g["act_off"], dtype=np.int64)) for g in groups])
    out = {k: np.concatenate([np.asarray(g[k]) for g in groups]) for k in OBS_ROW_KEYS
           if k != "tiles" and k in groups[0]}
    out["tiles"] = np.concatenate(tiles)
    out["board_mask"] = np.concatenate(masks)
    out["act_off"] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    out["actions"] = np.concatenate(actions)
    return out


# ---------------------------------------------------------------------------------------------------------------
# The network as a Policy (interfaces.md section 1.1)
# ---------------------------------------------------------------------------------------------------------------


class NetPolicy(ModelPolicy):
    """A `ModelPolicy` around a network held in memory instead of loaded from a file: the learner's, whose weights
    change after every update, and the frozen pool networks. Same `choose` (sampling, `logp`, `value`, `top`); the
    network is kept in eval mode, so BatchNorm uses its running statistics."""

    def __init__(self, model: PolyNet, policy_id: str, device: torch.device, max_batch: int = 256, seed: int = 0):
        self.path = None
        self.device = torch.device(device)
        self.max_batch = max_batch
        self.model = model.to(self.device).to(memory_format=torch.channels_last).eval()
        self.step = 0
        self.policy_id = policy_id
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.infer_seconds = 0.0
        self.decisions = 0


# ---------------------------------------------------------------------------------------------------------------
# Policy ids
# ---------------------------------------------------------------------------------------------------------------


def policy_id_of(path: Path, step: int) -> str:
    """`<run>@<step>`, the id interfaces.md stores with every move; the run is the checkpoint's directory name."""
    return f"{path.parent.name or 'ckpt'}@{step}"


# ---------------------------------------------------------------------------------------------------------------
# Opponent pool
# ---------------------------------------------------------------------------------------------------------------

DEFAULT_POOL_WEIGHTS = {
    "retail:easy": 0.05, "retail:normal": 0.15, "retail:hard": 0.15, "retail:crazy": 0.15,
    "bc": 0.15, "self": 0.35,
}


def parse_pairs(text: str, what: str) -> dict:
    """`a=1,b=2` -> {"a": 1.0, "b": 2.0}."""
    out = {}
    for item in text.split(","):
        if not item.strip():
            continue
        k, sep, v = item.partition("=")
        if not sep:
            raise ValueError(f"{what}: expected key=value, got {item!r}")
        out[k.strip()] = float(v)
    return out


def parse_players(text: str) -> dict[int, int]:
    """`2:55,3:25,4:20` -> {2: 55, 3: 25, 4: 20}; plain counts (`2,3`) get weight 1."""
    out: dict[int, int] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        k, sep, v = item.partition(":")
        n = int(k)
        if not 2 <= n <= MAX_SEATS:
            raise ValueError(f"player counts are 2 to {MAX_SEATS}, got {n}")
        out[n] = int(v) if sep else 1
    if not out or sum(out.values()) <= 0:
        raise ValueError("n_players needs at least one positive weight")
    return out


def parity_adjusted(mean: float, n_players: int) -> float:
    """Map a win rate to a scale where parity is 0.5 whatever the player count: chance is 1 / n, so 0.25 in a
    four-player game is as good as 0.5 one against one. Piecewise linear, 0 -> 0, chance -> 0.5, 1 -> 1."""
    c = 1.0 / n_players
    if mean <= c:
        return 0.5 * mean / c
    return 0.5 + 0.5 * (mean - c) / (1.0 - c)


class OppStats:
    """Running result of the learner against each pool entry, per player count, as exponentially decayed sums
    (`decay` per game of that entry). A win counts 1, a draw 0.5, a loss 0."""

    PRIOR = 5.0  # pseudo-games at parity, so a first win does not read as 100%

    def __init__(self, decay: float = 0.995):
        self.decay = decay
        self.cells: dict[tuple[str, int], list[float]] = {}  # (entry, n_players) -> [sum, count]

    def record(self, entry: str, n_players: int, outcome: float, weight: float = 1.0) -> None:
        """`weight` corrects the length bias of finished games: games end at a rate proportional to 1 / length, so a
        window of finished games over-represents short ones (mostly early losses). The trainer passes the game's turns
        over the running mean, which restores the per-game average."""
        cell = self.cells.setdefault((entry, n_players), [0.0, 0.0])
        cell[0] = cell[0] * self.decay + weight * outcome
        cell[1] = cell[1] * self.decay + weight

    def games(self, entry: str) -> float:
        return sum(c[1] for (e, _), c in self.cells.items() if e == entry)

    def rate(self, entry: str) -> float:
        """Parity-adjusted running win rate against `entry`; 0.5 with no games."""
        num = den = 0.0
        for (e, n), (s, cnt) in self.cells.items():
            if e != entry or cnt <= 0:
                continue
            c = 1.0 / n
            mean = (s + self.PRIOR * c) / (cnt + self.PRIOR)
            num += cnt * parity_adjusted(mean, n)
            den += cnt
        return num / den if den else 0.5

    def raw(self, entry: str, n_players: int) -> float | None:
        cell = self.cells.get((entry, n_players))
        return cell[0] / cell[1] if cell and cell[1] > 0 else None

    def state_dict(self) -> dict:
        return {"decay": self.decay, "cells": [[e, n, s, c] for (e, n), (s, c) in self.cells.items()]}

    def load_state_dict(self, d: dict) -> None:
        self.decay = d.get("decay", self.decay)
        self.cells = {(e, int(n)): [float(s), float(c)] for e, n, s, c in d["cells"]}


def pool_weights(entries: list[str], base: dict[str, float], stats: OppStats, beat_target: float = 0.7,
                 floor: float = 0.1, cap: float = 2.0) -> dict[str, float]:
    """Sampling weights over `entries`, summing to 1.

    The base weight of an entry (`self:<id>` entries share the weight of "self" equally) is multiplied by
    clip((1 - win rate) / (1 - beat_target), floor, cap): 1 at the target rate, more for opponents the learner beats
    less often, down to `floor` for those it always beats."""
    n_self = sum(1 for e in entries if e.startswith("self:"))
    raw = {}
    for e in entries:
        b = base.get("self", 0.0) / n_self if e.startswith("self:") else base.get(e, 0.0)
        hardness = min(cap, max(floor, (1.0 - stats.rate(e)) / (1.0 - beat_target)))
        raw[e] = b * hardness
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("every pool entry has weight 0")
    return {e: w / total for e, w in raw.items()}


# ---------------------------------------------------------------------------------------------------------------
# Games per arm
# ---------------------------------------------------------------------------------------------------------------
#
# VecEnv fixes the model seats and the per-seat handicaps for all of its games, so the games are grouped in "arms":
#   ("retail", entry, seat, shard)  one model seat (the learner) at `seat` with Normal handicap, every other seat
#                            the retail AI at that difficulty. Seat s only exists in games of more than s players.
#                            Split into shards of at most games / shards games, like the model arms, so that one
#                            heavily weighted retail entry does not leave a single thread stepping most of the games.
#   ("model", shard)         every seat is a model seat at Normal handicap; Python decides per game which seat is the
#                            learner and which frozen network plays the others. Sharded so that it steps on several
#                            threads.
# A seat of the learner is uniform over the seats of the game, so P(seat = s) = sum over n > s of w_n / n.


def seat_probs(n_players: dict[int, float]) -> list[float]:
    total = sum(n_players.values())
    return [sum(w / n for n, w in n_players.items() if n > s) / total for s in range(MAX_SEATS)]


def retail_player_weights(n_players: dict[int, float], seat: int) -> dict[int, int]:
    """Player-count weights of the arm whose learner sits at `seat`: P(n | seat) is proportional to w_n / n."""
    ws = {n: w / n for n, w in n_players.items() if n > seat}
    return {n: max(1, round(1000 * w)) for n, w in ws.items()}


def plan_quotas(weights: dict[str, float], games: int, seat_p: list[float], shards: int) -> dict[tuple, int]:
    """Active games per arm so that the pool entries get their weights: the expected counts rounded by largest
    remainder to `games` in total."""
    expect: dict[tuple, float] = {}
    per = games / shards  # games one arm holds at most, so that no arm steps far more games than the others
    for e, w in weights.items():
        if e.startswith("retail:"):
            for s, p in enumerate(seat_p):
                if p > 0:
                    v = games * w * p
                    n = max(1, math.ceil(v / per - 1e-9))
                    for k in range(n):
                        expect[("retail", e, s, k)] = v / n
    model_w = sum(w for e, w in weights.items() if not e.startswith("retail:"))
    for k in range(shards):
        expect[("model", k)] = games * model_w / shards
    quota = {k: int(math.floor(v)) for k, v in expect.items()}
    short = games - sum(quota.values())
    order = sorted(expect, key=lambda k: (-(expect[k] - quota[k]), str(k)))
    for k in order[:max(short, 0)]:
        quota[k] += 1
    return quota


def arm_capacity(key: tuple, games: int, seat_p: list[float], shards: int) -> int:
    """Games an arm is built with: enough for it to carry the whole pool weight (plus rounding slack)."""
    if key[0] == "retail":
        return min(math.ceil(games * seat_p[key[2]]), math.ceil(games / shards)) + 2
    return math.ceil(games / shards) + 2


@dataclass
class EntryInfo:
    """A pool entry as logged: what it is and which checkpoint plays it."""

    entry: str
    policy_id: str
    path: str = ""
    extra: dict = field(default_factory=dict)
