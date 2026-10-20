"""Checks against the Rust side: the `polyai_engine` constants and real shards written by `polyai-gen`.

Both skip when the thing they need is absent (the engine module is not installed, the smoke shards are not there).
"""

from pathlib import Path

import numpy as np
import pytest

from polyai import schema as S
from polyai.data import BatchLoader, Shard, tile_permutation, transform_actions, transform_tiles, valid_symmetries

SMOKE = Path("/kvm/polytopia-ai-data/smoke")
COL = S.COLUMN


def test_schema_matches_the_engine_constants():
    pe = pytest.importorskip("polyai_engine")
    assert pe.ENCODING_VERSION == S.ENCODING_VERSION
    assert pe.F_TILE == S.F_TILE
    assert pe.G_DIM == S.G_DIM
    assert pe.N_KINDS == S.N_KINDS
    assert list(pe.TRIBES) == S.TRIBES
    assert list(pe.CAT_VOCABS) == S.CAT_VOCABS
    assert list(pe.KIND_NAMES) == S.ACTION_KINDS


def smoke_shards() -> list[Path]:
    return sorted(SMOKE.glob("*.npz")) if SMOKE.is_dir() else []


@pytest.fixture(scope="module")
def real() -> Shard:
    paths = smoke_shards()
    if not paths:
        pytest.skip("no smoke shards in /kvm/polytopia-ai-data/smoke")
    return Shard.load(min(paths, key=lambda p: p.stat().st_size))  # the smallest file loads fastest


def test_real_shard_loads_and_every_chosen_is_in_its_legal_list(real):
    a = real.a
    lens = np.diff(a["act_off"])
    assert (lens >= 1).all()
    assert ((a["chosen"] >= 0) & (a["chosen"] < lens)).all()
    cand = a["cand_idx"]
    assert ((cand < lens[:, None]) & (cand >= -1)).all()
    assert (a["winner"] >= 0).all() and (a["winner"] < S.N_SLOTS).all()
    assert (a["actions"][:, 1] >= 0).all() and (a["actions"][:, 1] < S.ARG_MAX).all()
    assert (a["actions"][:, 0] >= 1).all() and (a["actions"][:, 0] <= len(S.ACTION_KINDS)).all()
    for i, v in enumerate(S.CAT_VOCABS):
        assert a["tiles"][..., i].max() < v


def test_real_candidates_are_a_best_first_prefix_with_zero_padding(real):
    c, s = real.a["cand_idx"], real.a["cand_score"]
    valid = c >= 0
    assert (s[~valid] == 0).all()
    assert not np.isnan(s).any()
    assert (valid[:, 1:] <= valid[:, :-1]).all()  # valid entries come first
    capped = np.minimum(s, 1e30)  # +inf is allowed and ranks first
    both = valid[:, 1:]
    assert (np.diff(capped, axis=1)[both] <= 1e-3).all()


def test_real_action_tiles_hold_what_the_kind_expects(real):
    """The semantic anchor that makes the symmetry test meaningful: Train acts on a city of mine, Move on my unit."""
    a = real.a
    n = real.n
    t = a["tiles"].reshape(n, real.h * real.w, S.F_TILE)
    ch = a["actions"][a["act_off"][:-1] + a["chosen"]]
    rows = np.arange(n)
    train = ch[:, 0] == S.KIND_ID["Train"]
    assert train.any()
    src = ch[train, 2]
    assert (t[rows[train], src, COL["is_city"]] == 1).all()
    assert (t[rows[train], src, COL["owner"]] == 1).all()
    move = ch[:, 0] == S.KIND_ID["Move"]
    src = ch[move, 2]
    assert (t[rows[move], src, COL["unit_type"]] > 0).all()
    assert (t[rows[move], src, COL["unit_owner"]] == 1).mean() > 0.99


@pytest.mark.parametrize("sym", range(8))
def test_symmetry_keeps_chosen_actions_on_the_same_tile_contents(real, sym):
    if sym not in valid_symmetries(real.h, real.w):
        pytest.skip("rectangular board")
    a = real.a
    idx = np.arange(0, real.n, 7)[:400]
    t0 = a["tiles"][idx]
    t1 = transform_tiles(t0, sym)
    act0 = a["actions"][a["act_off"][idx] + a["chosen"][idx]]
    act1 = transform_actions(act0, real.h, real.w, sym)
    f0 = t0.reshape(len(idx), real.h * real.w, S.F_TILE)
    f1 = t1.reshape(len(idx), real.h * real.w, S.F_TILE)
    rows = np.arange(len(idx))
    for col in (2, 3):
        has = act0[:, col] >= 0
        np.testing.assert_array_equal(f1[rows[has], act1[has, col]], f0[rows[has], act0[has, col]])
        assert (act1[~has, col] == -1).all()
    assert (act1[:, :2] == act0[:, :2]).all()
    train = act1[:, 0] == S.KIND_ID["Train"]
    assert train.any()
    assert (f1[rows[train], act1[train, 2], COL["is_city"]] == 1).all()  # still a city after the move


def test_augmented_real_batch_keeps_every_legal_list_intact(real):
    b = real.batch(np.arange(64), np.random.default_rng(0), augment=True)
    a = real.a
    for row in range(64):
        n = a["act_off"][row + 1] - a["act_off"][row]
        assert b["action_mask"][row].sum() == n
        assert b["chosen"][row] < n
        src = b["actions"][row, :n, 2]
        assert ((src >= -1) & (src < real.h * real.w)).all()
        np.testing.assert_array_equal(b["actions"][row, :n, :2], a["actions"][a["act_off"][row] : a["act_off"][row + 1], :2])


def test_loader_batches_real_shards_one_map_size_at_a_time():
    paths = smoke_shards()
    if not paths:
        pytest.skip("no smoke shards")
    sizes = set()
    for p in paths:
        batch = next(iter(BatchLoader([p], 32, prefetch=0, shuffle=False).iterate(epochs=1)))
        assert batch["tiles"].shape[1] == batch["tiles"].shape[2]
        assert (batch["chosen"] < batch["action_mask"].sum(1)).all()
        assert batch["globals"].shape == (32, S.G_DIM)
        sizes.add(batch["tiles"].shape[1])
    assert sizes <= {11, 14, 16}


def test_corners_stay_corners_under_every_symmetry(real):
    h, w = real.h, real.w
    corners = {0, w - 1, (h - 1) * w, h * w - 1}
    for sym in valid_symmetries(h, w):
        _, remap = tile_permutation(h, w, sym)
        assert {int(remap[c]) for c in corners} == corners
