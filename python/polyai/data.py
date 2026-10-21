"""Shard loading, batching by map size, 8-fold symmetry augmentation and synthetic shards.

Shard layout is `schema.SHARD_FIELDS` (encoding v2). Version 1 shards are read too and relabelled to the v2 meaning:
games that ended by top-score adjudication become draws (`winner = -1`, `capped = 1`), the score margins are dropped and
the "turns left" global, which only told the model about a 50-turn limit, is zeroed as v2 does for Domination.
One shard holds one map size, so every batch drawn from a shard has a single
(H, W) and needs no board padding. Only the legal-action lists are padded.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from . import schema as S

# ---------------------------------------------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------------------------------------------

N_SYMMETRIES = 8


def _apply_symmetry(grid: np.ndarray, sym: int) -> np.ndarray:
    """Symmetry `sym` = 4 * flip + rot: flip the x axis first when flip is 1, then rotate by `rot` quarter turns."""
    rot, flip = sym % 4, sym // 4
    if flip:
        grid = grid[:, ::-1]
    return np.rot90(grid, rot)


def valid_symmetries(h: int, w: int) -> list[int]:
    """All 8 on a square board; on a rectangle only those that keep the shape (even quarter turns)."""
    return list(range(N_SYMMETRIES)) if h == w else [s for s in range(N_SYMMETRIES) if s % 2 == 0]


@lru_cache(maxsize=None)
def tile_permutation(h: int, w: int, sym: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (gather, remap) for a board of h x w tiles.

    gather[new_flat] = old_flat, so `new_tiles = old_tiles.reshape(h * w, ...)[gather]` (the new board has shape
    (h', w'), flat index y' * w' + x').
    remap[old_flat] = new_flat, which is what an action's src/dst tile index goes through.
    """
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    gather = np.ascontiguousarray(_apply_symmetry(idx, sym)).reshape(-1)
    remap = np.empty_like(gather)
    remap[gather] = np.arange(h * w, dtype=np.int64)
    return gather, remap


def transform_tiles(tiles: np.ndarray, sym: int) -> np.ndarray:
    """tiles [..., H, W, F] -> transformed copy."""
    h, w = tiles.shape[-3], tiles.shape[-2]
    gather, _ = tile_permutation(h, w, sym)
    nh, nw = (h, w) if sym % 2 == 0 else (w, h)
    flat = tiles.reshape(*tiles.shape[:-3], h * w, tiles.shape[-1])
    return flat[..., gather, :].reshape(*tiles.shape[:-3], nh, nw, tiles.shape[-1])


def transform_actions(actions: np.ndarray, h: int, w: int, sym: int) -> np.ndarray:
    """actions [..., 4] = (kind, arg, src, dst); tile indices (>= 0) follow the symmetry, -1 stays -1."""
    _, remap = tile_permutation(h, w, sym)
    out = actions.copy()
    for col in (2, 3):
        t = out[..., col].astype(np.int64)
        out[..., col] = np.where(t >= 0, remap[np.clip(t, 0, h * w - 1)], -1)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------------------------------------------


def save_shard(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    missing = [k for k in S.REQUIRED_FIELDS if k not in arrays]
    if missing:
        raise ValueError(f"shard is missing fields: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(arrays[k], dtype=S.SHARD_FIELDS[k].dtype) for k in S.REQUIRED_FIELDS})


def relabel_v1(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """A version 1 shard in the version 2 meaning (see the module docstring). Does not modify its input."""
    adjudicated = np.asarray(arrays["adjudicated"]).astype(np.int8)
    out = {k: arrays[k] for k in S.REQUIRED_FIELDS if k in arrays}
    out["winner"] = np.where(adjudicated == 1, -1, arrays["winner"]).astype(np.int8)
    out["capped"] = (adjudicated == 1).astype(np.int8)
    g = np.array(arrays["globals"], dtype=np.float32)
    g[:, S.G_TURNS_LEFT] = 0.0
    out["globals"] = g
    out["version"] = np.array([S.ENCODING_VERSION], np.int32)
    return out


class Shard:
    """One loaded shard held in memory, always in the version 2 meaning (`source_version` says what was read)."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        if "version" not in arrays:
            raise ValueError("shard has no version field")
        version = int(np.asarray(arrays["version"]).reshape(-1)[0])
        if version not in S.SUPPORTED_VERSIONS:
            raise ValueError(f"shard has encoding version {version}, this package reads {list(S.SUPPORTED_VERSIONS)}")
        required = S.V1_REQUIRED_FIELDS if version == 1 else S.REQUIRED_FIELDS
        missing = [k for k in required if k not in arrays]
        if missing:
            raise ValueError(f"version {version} shard is missing fields: {missing}")
        if version == 1:
            arrays = relabel_v1(arrays)
        self.source_version = version
        self.a = arrays
        n, h, w, f = arrays["tiles"].shape
        if f != S.F_TILE:
            raise ValueError(f"tiles have {f} fields, schema says {S.F_TILE}")
        if arrays["globals"].shape != (n, S.G_DIM):
            raise ValueError(f"globals have shape {arrays['globals'].shape}, schema says ({n}, {S.G_DIM})")
        if arrays["act_off"].shape != (n + 1,):
            raise ValueError("act_off must have N + 1 entries")
        if arrays["actions"].shape != (int(arrays["act_off"][-1]), 4):
            raise ValueError("actions must have act_off[-1] rows of 4")
        if ((arrays["capped"] == 1) & (arrays["winner"] >= 0)).any():
            raise ValueError("a capped game has a winner")
        self.n, self.h, self.w = n, h, w

    @classmethod
    def load(cls, path: str | Path) -> Shard:
        with np.load(path) as z:
            if "version" not in z.files:
                raise ValueError(f"{path} has no version field")
            wanted = S.V1_REQUIRED_FIELDS if int(z["version"].reshape(-1)[0]) == 1 else S.REQUIRED_FIELDS
            return cls({k: z[k] for k in wanted if k in z.files})

    def __len__(self) -> int:
        return self.n

    def batch(self, idx: np.ndarray, rng: np.random.Generator | None = None, augment: bool = False) -> dict:
        """Gather positions `idx` into a dict of numpy arrays. With `augment`, each position gets its own random
        symmetry applied to its tiles and to the src/dst tiles of its actions."""
        a = self.a
        idx = np.asarray(idx)
        b = len(idx)
        starts = a["act_off"][idx].astype(np.int64)
        lens = (a["act_off"][idx + 1] - a["act_off"][idx]).astype(np.int64)
        amax = max(int(lens.max()), 1)
        cols = np.arange(amax)[None, :]
        mask = cols < lens[:, None]
        rows = np.where(mask, starts[:, None] + cols, 0)
        actions = a["actions"][rows].astype(np.int16)
        actions[~mask] = (S.KIND_PAD, 0, -1, -1)

        tiles = a["tiles"][idx]
        if augment:
            if rng is None:
                rng = np.random.default_rng()
            allowed = np.array(valid_symmetries(self.h, self.w))
            syms = allowed[rng.integers(0, len(allowed), size=b)]
            for s in np.unique(syms):
                if s == 0:
                    continue
                sel = np.nonzero(syms == s)[0]
                tiles[sel] = transform_tiles(tiles[sel], int(s))  # odd turns only occur on square boards
                actions[sel] = transform_actions(actions[sel], self.h, self.w, int(s))

        return {
            "tiles": tiles,
            "globals": a["globals"][idx],
            "actions": actions,
            "action_mask": mask,
            "chosen": a["chosen"][idx].astype(np.int64),
            "cand_idx": a["cand_idx"][idx].astype(np.int64),
            "cand_score": a["cand_score"][idx],  # may hold +inf; compute_losses deals with it
            "winner": a["winner"][idx].astype(np.int64),
            "turn": a["turn"][idx].astype(np.int64),
        }


def to_torch(batch: dict, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True) for k, v in batch.items()}


class BatchLoader:
    """Iterates the shards in a (seeded) random order and yields batches from one shard at a time, so every batch
    has a single map size. A background thread prepares the next batches while the GPU is busy.

    `epochs=None` loops forever. A final partial batch is dropped when `drop_last` is set.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        batch_size: int,
        *,
        augment: bool = True,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        prefetch: int = 4,
        device: torch.device | str = "cpu",
    ):
        if not paths:
            raise ValueError("no shards given")
        self.paths = [Path(p) for p in paths]
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.prefetch = prefetch
        self.device = device

    def _numpy_batches(self, epochs: int | None) -> Iterator[dict]:
        rng = np.random.default_rng(self.seed)
        epoch = 0
        while epochs is None or epoch < epochs:
            order = rng.permutation(len(self.paths)) if self.shuffle else np.arange(len(self.paths))
            for pi in order:
                shard = Shard.load(self.paths[pi])
                idx = rng.permutation(shard.n) if self.shuffle else np.arange(shard.n)
                for s in range(0, shard.n, self.batch_size):
                    sel = idx[s : s + self.batch_size]
                    if len(sel) < self.batch_size and self.drop_last:
                        continue
                    yield shard.batch(sel, rng, self.augment)
            epoch += 1

    def iterate(self, epochs: int | None = 1) -> Iterator[dict[str, torch.Tensor]]:
        if self.prefetch <= 0:
            for b in self._numpy_batches(epochs):
                yield to_torch(b, self.device)
            return
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()
        done = object()

        def work() -> None:
            try:
                for b in self._numpy_batches(epochs):
                    item = to_torch(b, "cpu")
                    while not stop.is_set():
                        try:
                            q.put(item, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    if stop.is_set():
                        return
                q.put(done)
            except BaseException as e:  # surfaced in the consumer
                q.put(e)

        t = threading.Thread(target=work, daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is done:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield {k: v.to(self.device, non_blocking=True) for k, v in item.items()}
        finally:
            stop.set()


# ---------------------------------------------------------------------------------------------------------------
# Synthetic shards, same shapes as real ones. Swap for real shards by pointing the loader at them.
# ---------------------------------------------------------------------------------------------------------------


def make_synthetic_shard(
    n: int, size: int = 16, seed: int = 0, n_cand: int = S.N_CANDIDATES, max_actions: int = 40
) -> dict:
    """Random positions with a learnable structure: the chosen action is the legal one with the highest hidden score
    that depends on its kind, argument and the terrain of its source tile; the winner is a function of the globals (-1 for the few capped games).
    Like the real data, the candidate list holds only some of the positions' top actions, `chosen` is sometimes not
    its first entry, some scores are +inf and padding scores are 0."""
    rng = np.random.default_rng(seed)
    h = w = size
    tiles = np.zeros((n, h, w, S.F_TILE), np.int16)
    for i, v in enumerate(S.CAT_VOCABS):
        tiles[..., i] = rng.integers(0, min(v, 6), size=(n, h, w))
    for j, (name, scale) in enumerate(S.NUM_FIELDS):
        col = S.N_CAT + j
        hi = 2 if scale == 1.0 else int(scale)
        tiles[..., col] = rng.integers(0, hi + 1, size=(n, h, w))
    globals_ = rng.normal(size=(n, S.G_DIM)).astype(np.float32)

    lens = rng.integers(2, max_actions + 1, size=n)
    off = np.zeros(n + 1, np.int32)
    off[1:] = np.cumsum(lens)
    total = int(off[-1])
    kinds = rng.integers(1, len(S.ACTION_KINDS) + 1, size=total)
    args = rng.integers(0, 12, size=total)  # real args reach 62 (unit and improvement idx); the embedding covers 63
    src = rng.integers(0, h * w, size=total)
    dst = rng.integers(0, h * w, size=total)
    pos_of = np.repeat(np.arange(n), lens)
    no_tile = rng.random(total) < 0.15
    src = np.where(no_tile, -1, src)
    dst = np.where(no_tile, -1, dst)
    actions = np.stack([kinds, args, src, dst], axis=1).astype(np.int16)

    terrain_at_src = tiles[pos_of, np.clip(src, 0, h * w - 1) // w, np.clip(src, 0, h * w - 1) % w, 0]
    hidden = (kinds * 7 % 11) + 0.5 * (args % 5) + 0.8 * terrain_at_src * (src >= 0) + rng.normal(0, 0.05, total)

    chosen = np.zeros(n, np.int16)
    cand_idx = -np.ones((n, n_cand), np.int16)
    cand_score = np.zeros((n, n_cand), np.float32)
    for i in range(n):
        s = hidden[off[i] : off[i + 1]]
        order = np.argsort(-s)
        chosen[i] = order[0] if i % 4 else order[min(1, len(order) - 1)]  # the played move is not always the best
        if i % 3:  # a third of the positions carry no candidate list
            k = min(n_cand, len(order))
            cand_idx[i, :k] = order[:k]
            cand_score[i, :k] = s[order[:k]] * 20.0
            if i % 7 == 0:
                cand_score[i, 0] = np.inf

    winner = np.where(globals_[:, 0] > 0, 0, 1 + (globals_[:, 1] > 0)).astype(np.int8)
    capped = (rng.random(n) < 0.05).astype(np.int8)
    winner[capped == 1] = -1  # a capped game is a draw
    turn = rng.integers(1, 50, size=n).astype(np.int16)
    final_scores = np.zeros((n, 8), np.float32)
    final_scores[:, 0] = np.abs(globals_[:, 2]) * 3000
    return {
        "tiles": tiles, "globals": globals_, "act_off": off, "actions": actions, "chosen": chosen,
        "cand_idx": cand_idx, "cand_score": cand_score, "winner": winner, "final_scores": final_scores,
        "capped": capped, "turn": turn,
        "seed": rng.integers(1, 100000, size=n).astype(np.int32), "cmd_index": np.arange(n, dtype=np.int32),
        "difficulty": rng.integers(0, 4, size=n).astype(np.int8), "version": np.array([S.ENCODING_VERSION], np.int32),
    }


def write_synthetic_shards(
    out_dir: str | Path, n_shards: int, n: int, size: int = 16, seed: int = 0, **kw
) -> list[Path]:
    out = Path(out_dir)
    paths = []
    for k in range(n_shards):
        p = out / f"synthetic_{size}_{seed}_{k:04d}.npz"
        save_shard(p, make_synthetic_shard(n, size, seed * 1000 + k, **kw))
        paths.append(p)
    return paths
