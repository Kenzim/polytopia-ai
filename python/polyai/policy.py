"""The policy interface (docs/interfaces.md section 1.1) and `ModelPolicy`, a checkpoint behind it.

Everything that wants moves from a trained model (the arena, self-play, the socket service in `polyai.serve`) goes
through `Policy.choose`. The input is an observation group of the engine module: the dict `VecEnv.observe()` and
`VecEnv.step()` return for one map size (`tiles`, `globals`, `act_off`, `actions`, ...), or the same built from single
games with `observe_games`.

`Choices.logp` and the probabilities in `Choices.top` are those of the distribution the pick was drawn from:
`softmax(logits / T)` for a sampled move, and the plain `softmax(logits)` (T = 1) for a greedy one. So a greedy move
reports the model's own probability of it (not 1), which is what a log wants and what an importance ratio against a
later checkpoint needs. `value` is the winner head's probability that the player to move wins by elimination (slot 0
of the winner softmax); `polyai.model.eval_pawns(value)` turns it into the number shown to people.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from . import schema as S

TOP_K = 3
MAX_BATCH = 128


@dataclass
class Choices:
    """One entry per position of the observation."""

    index: np.ndarray  # int32[N], index into that position's legal list
    logp: np.ndarray  # float32[N], log-probability of `index` (see the module docstring)
    value: np.ndarray  # float32[N], P(the mover wins) from the value head
    top: list[list[tuple[int, float]]]  # per position, the best three (index, probability), best first

    def __len__(self) -> int:
        return len(self.index)


@runtime_checkable
class Policy(Protocol):
    policy_id: str  # "<run>@<step>", e.g. "bc4_b10c128_v2@100000"; stored with every move

    def choose(self, obs: dict, *, temperature: float | None = None) -> Choices:
        """A move for every position of `obs`. `temperature=None` is the greedy move; a positive value samples from
        softmax(logits / T)."""
        ...


# ---------------------------------------------------------------------------------------------------------------
# Observations and batching
# ---------------------------------------------------------------------------------------------------------------


def _legal_rows(obs: dict, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rows, mask) of positions `sel`: the padded [B, A] index into the flat legal-action arrays and which entries are real."""
    act_off = np.asarray(obs["act_off"], dtype=np.int64)
    starts, lens = act_off[:-1][sel], np.diff(act_off)[sel]
    amax = max(int(lens.max()), 1)
    cols = np.arange(amax)[None, :]
    mask = cols < lens[:, None]
    return np.where(mask, starts[:, None] + cols, 0), mask


def collate_features(obs: dict, sel: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    """The optional engine features (`f_tiles`, `f_globals`, `f_actions`, encoding.md section 12) of positions `sel`,
    laid out like `collate` does (`f_actions` padded with zeros to [B, A, FEAT_A]). Empty when `obs` has none."""
    if "f_tiles" not in obs:
        return {}
    rows, mask = _legal_rows(obs, sel)
    f_actions = np.zeros((*mask.shape, S.FEAT_A), np.float32)
    f_actions[mask] = np.asarray(obs["f_actions"])[rows[mask]]
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True) for k, v in (
        ("f_tiles", np.asarray(obs["f_tiles"])[sel]), ("f_globals", np.asarray(obs["f_globals"])[sel]),
        ("f_actions", f_actions))}


def collate(obs: dict, sel: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Positions `sel` of an observation group (one map size, or padded boards with a `board_mask`, see
    `rl_pool.merge_padded`) -> tiles, globals, padded actions, action mask."""
    rows, mask = _legal_rows(obs, sel)
    actions = np.asarray(obs["actions"])[rows].astype(np.int16)
    actions[~mask] = (S.KIND_PAD, 0, -1, -1)

    def put(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a)).to(device, non_blocking=True)

    return put(np.asarray(obs["tiles"])[sel]), put(np.asarray(obs["globals"])[sel]), put(actions), put(mask)


def observe_games(games: list) -> dict:
    """An observation group from single `polyai_engine.Game`s of one map size, each awaiting a decision."""
    if not games:
        raise ValueError("no games")
    shapes = {g.shape for g in games}
    if len(shapes) != 1:
        raise ValueError(f"games of different map sizes cannot share an observation: {sorted(shapes)}")
    tiles, glob, acts = [], [], []
    for g in games:
        t, gl = g.encode()
        tiles.append(t)
        glob.append(gl)
        acts.append(g.legal_actions())
    off = np.concatenate([[0], np.cumsum([len(a) for a in acts])]).astype(np.int32)
    return {
        "game": np.arange(len(games), dtype=np.int32),
        "seat": np.array([g.current_seat() or 0 for g in games], dtype=np.int8),
        "n_players": np.array([g.n_players for g in games], dtype=np.int8),
        "turn": np.array([g.turn for g in games], dtype=np.int16),
        "tiles": np.stack(tiles),
        "globals": np.stack(glob),
        "act_off": off,
        "actions": np.concatenate(acts) if acts else np.zeros((0, 4), np.int16),
    }


def symmetric_views(obs: dict) -> list[dict]:
    """`obs` under every board symmetry that keeps its shape (test-time augmentation): tiles, engine tile features
    and the actions' src/dst tile indices are transformed together; the legal list keeps its order."""
    from .data import transform_actions, transform_tiles, valid_symmetries

    tiles = np.asarray(obs["tiles"])
    h, w = tiles.shape[-3], tiles.shape[-2]
    views = []
    for sym in valid_symmetries(h, w):
        if sym % 2 and h != w:
            continue
        view = dict(obs)
        view["tiles"] = transform_tiles(tiles, sym)
        view["actions"] = transform_actions(np.asarray(obs["actions"]), h, w, sym)
        if "f_tiles" in obs:
            view["f_tiles"] = transform_tiles(np.asarray(obs["f_tiles"]), sym)
        views.append(view)
    return views


# ---------------------------------------------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------------------------------------------


def resolve_checkpoint(path: str | Path) -> Path:
    """A checkpoint file, or a run directory (then its `ckpt_final.pt`, else `ckpt_last.pt`)."""
    path = Path(path)
    if path.is_dir():
        for name in ("ckpt_final.pt", "ckpt_last.pt"):
            if (path / name).exists():
                return path / name
        raise FileNotFoundError(f"no ckpt_final.pt or ckpt_last.pt in {path}")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint {path} does not exist")
    return path


class ModelPolicy:
    """A checkpoint as a `Policy`: batched inference (bf16 autocast on CUDA), greedy or temperature sampling.

    `policy_id` is `<run>@<step>`: the run directory's name and the step the checkpoint was written at. Observations
    larger than `max_batch` positions go through the network in chunks. `seed` seeds the sampling generator.
    `infer_seconds` and `decisions` count the time spent inside `choose` and the positions answered.
    """

    def __init__(self, path: str | Path, device: str | torch.device = "cpu", max_batch: int = MAX_BATCH, seed: int = 0,
                 tta: bool = False):
        from .train_bc import load_checkpoint

        self.tta = tta

        self.path = resolve_checkpoint(path)
        self.device = torch.device(device)
        self.max_batch = max_batch
        self.model = load_checkpoint(self.path).to(self.device).to(memory_format=torch.channels_last).eval()
        state = torch.load(self.path, map_location="cpu", weights_only=False)
        self.step = int(state.get("step", 0))
        self.encoding_version = int(state.get("encoding_version", S.ENCODING_VERSION))
        self.run = self.path.parent.name or "ckpt"
        self.policy_id = f"{self.run}@{self.step}"
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.infer_seconds = 0.0
        self.decisions = 0

    @property
    def features(self) -> bool:
        """Whether the network reads the engine features (the observations then need `VecEnv(features=True)`)."""
        return bool(self.model.cfg.features)

    @torch.inference_mode()
    def choose(self, obs: dict, *, temperature: float | None = None) -> Choices:
        if temperature is not None and not temperature > 0:
            raise ValueError("the sampling temperature must be positive (None is the greedy move)")
        act_off = np.asarray(obs["act_off"], dtype=np.int64)
        n = len(act_off) - 1
        lens = np.diff(act_off)
        if (lens <= 0).any():
            raise ValueError("a position has an empty legal list")
        index = np.empty(n, np.int32)
        logp = np.empty(n, np.float32)
        value = np.empty(n, np.float32)
        top: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        t0 = time.perf_counter()
        board = obs.get("board_mask")
        views = symmetric_views(obs) if self.tta and board is None else [obs]
        for s in range(0, n, self.max_batch):
            sel = np.arange(s, min(n, s + self.max_batch))
            logits_sum, win_sum = None, None
            for view in views:
                tiles, globals_, actions, mask = collate(view, sel, self.device)
                feats = collate_features(view, sel, self.device) if self.model.cfg.features else {}
                if self.model.cfg.features and not feats:
                    raise ValueError(f"{self.policy_id} reads the engine features, but the observation has none "
                                     "(build the VecEnv with features=True)")
                bmask = None
                if board is not None:
                    bmask = torch.from_numpy(np.ascontiguousarray(np.asarray(board)[sel])).to(self.device, non_blocking=True)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    out = self.model(tiles, globals_, actions, mask, bmask, **feats)
                lp = torch.log_softmax(out.policy_logits.float(), dim=1)
                wp = torch.softmax(out.winner_logits.float(), dim=-1)
                logits_sum = lp if logits_sum is None else logits_sum + lp
                win_sum = wp if win_sum is None else win_sum + wp
            # with one view these are the model's own logits up to a per-row constant, so argmax and softmax agree
            logits = logits_sum / len(views)
            logits = logits.masked_fill(~mask, float("-inf"))
            win_probs = win_sum / len(views)
            if temperature is None:
                pick = logits.argmax(dim=1)
                log_probs = torch.log_softmax(logits, dim=1)
            else:
                pick = torch.multinomial(torch.softmax(logits / temperature, dim=1), 1, generator=self.generator)[:, 0]
                log_probs = torch.log_softmax(logits / temperature, dim=1)
            win = win_probs[:, 0]
            k = min(TOP_K, logits.shape[1])
            top_lp, top_ix = log_probs.topk(k, dim=1)
            # one device-to-host copy for everything (each .cpu() is a synchronisation)
            packed = torch.cat([pick[:, None].float(), log_probs.gather(1, pick[:, None]), win[:, None],
                                top_lp.exp(), top_ix.float()], dim=1).cpu().numpy()
            index[sel] = packed[:, 0].astype(np.int32)
            logp[sel] = packed[:, 1]
            value[sel] = packed[:, 2]
            top_p, top_i = packed[:, 3:3 + k], packed[:, 3 + k:].astype(np.int64)
            for row, i in enumerate(sel):
                top[i] = [(int(top_i[row, j]), float(top_p[row, j])) for j in range(k) if top_i[row, j] < lens[i]]
        self.infer_seconds += time.perf_counter() - t0
        self.decisions += n
        return Choices(index, logp, value, top)
