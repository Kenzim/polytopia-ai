import numpy as np
import pytest

from polyai import schema as S
from polyai.data import (
    BatchLoader,
    N_SYMMETRIES,
    Shard,
    make_synthetic_shard,
    save_shard,
    tile_permutation,
    transform_actions,
    transform_tiles,
    valid_symmetries,
)


def test_schema_dimensions():
    assert S.F_TILE == S.N_CAT + S.N_NUM == len(S.COLUMN)
    assert len(S.KIND_ID) == len(S.ACTION_KINDS) < S.N_KINDS
    assert S.G_DIM == 13 + 8 + 49 + 7 * 23 + 8


def test_synthetic_shard_matches_schema():
    arrays = make_synthetic_shard(32, 11, seed=3)
    shard = Shard(arrays)
    assert (shard.n, shard.h, shard.w) == (32, 11, 11)
    for name, spec in S.SHARD_FIELDS.items():
        assert arrays[name].dtype == np.dtype(spec.dtype), name
    for i, v in enumerate(S.CAT_VOCABS):
        assert arrays["tiles"][..., i].max() < v


def test_save_and_load_roundtrip(tmp_path):
    arrays = make_synthetic_shard(16, 11, seed=4)
    save_shard(tmp_path / "s.npz", arrays)
    loaded = Shard.load(tmp_path / "s.npz")
    for k in S.REQUIRED_FIELDS:
        np.testing.assert_array_equal(loaded.a[k], arrays[k])


def test_missing_field_is_rejected(tmp_path):
    arrays = make_synthetic_shard(4, 11)
    del arrays["margin5"]
    with pytest.raises(ValueError, match="margin5"):
        save_shard(tmp_path / "s.npz", arrays)


def test_batch_padding_and_mask(shard16):
    idx = np.array([0, 5, 9])
    b = shard16.batch(idx)
    a = shard16.a
    for row, i in enumerate(idx):
        n = a["act_off"][i + 1] - a["act_off"][i]
        assert b["action_mask"][row].sum() == n
        np.testing.assert_array_equal(b["actions"][row, :n], a["actions"][a["act_off"][i] : a["act_off"][i + 1]])
        assert (b["actions"][row, n:, 0] == S.KIND_PAD).all()
        assert (b["actions"][row, n:, 2:] == -1).all()
        assert b["chosen"][row] < n


@pytest.mark.parametrize("sym", range(N_SYMMETRIES))
def test_permutation_is_a_bijection_and_inverse(sym):
    gather, remap = tile_permutation(7, 7, sym)
    assert sorted(gather) == list(range(49))
    np.testing.assert_array_equal(remap[gather], np.arange(49))
    np.testing.assert_array_equal(gather[remap], np.arange(49))


def test_the_eight_symmetries_are_distinct():
    seen = {tuple(tile_permutation(5, 5, s)[0]) for s in range(N_SYMMETRIES)}
    assert len(seen) == 8


def test_rectangles_only_get_shape_preserving_symmetries():
    assert valid_symmetries(5, 5) == list(range(8))
    assert valid_symmetries(4, 6) == [0, 2, 4, 6]


@pytest.mark.parametrize("sym", range(N_SYMMETRIES))
def test_actions_follow_their_tiles(shard16, sym):
    """The tile content at an action's transformed src/dst equals the content at the original src/dst."""
    h = w = shard16.h
    b = shard16.batch(np.arange(16))
    tiles, actions = b["tiles"], b["actions"]
    new_tiles = transform_tiles(tiles, sym)
    new_actions = transform_actions(actions, h, w, sym)
    flat_old = tiles.reshape(16, h * w, -1)
    flat_new = new_tiles.reshape(16, h * w, -1)
    checked = 0
    for col in (2, 3):
        old_t, new_t = actions[..., col], new_actions[..., col]
        np.testing.assert_array_equal(old_t < 0, new_t < 0)
        for i in range(16):
            real = old_t[i] >= 0
            np.testing.assert_array_equal(flat_old[i, old_t[i][real]], flat_new[i, new_t[i][real]])
            checked += int(real.sum())
    assert checked > 100
    np.testing.assert_array_equal(new_actions[..., :2], actions[..., :2])


def test_geometry_of_known_symmetries():
    h = w = 4
    tile = lambda y, x: y * w + x  # noqa: E731
    act = np.array([[1, 0, tile(0, 1), tile(2, 3)]], np.int16)
    # sym 1 is one quarter turn (np.rot90: counter-clockwise): (y, x) -> (w - 1 - x, y)
    out = transform_actions(act, h, w, 1)[0]
    assert out[2] == tile(w - 1 - 1, 0) and out[3] == tile(w - 1 - 3, 2)
    # sym 4 mirrors x: (y, x) -> (y, w - 1 - x)
    out = transform_actions(act, h, w, 4)[0]
    assert out[2] == tile(0, 2) and out[3] == tile(2, 0)


def test_every_symmetry_has_exactly_one_inverse(shard16):
    """Applying a symmetry and then some other one returns the original tiles and actions, for exactly one other."""
    h = w = shard16.h
    b = shard16.batch(np.arange(8))
    for s in range(N_SYMMETRIES):
        t = transform_tiles(b["tiles"], s)
        a = transform_actions(b["actions"], h, w, s)
        back = [
            r for r in range(N_SYMMETRIES)
            if np.array_equal(transform_tiles(t, r), b["tiles"])
            and np.array_equal(transform_actions(a, h, w, r), b["actions"])
        ]
        assert len(back) == 1, (s, back)


def test_augmented_batch_keeps_everything_else(shard16):
    idx = np.arange(32)
    plain = shard16.batch(idx)
    aug = shard16.batch(idx, np.random.default_rng(5), augment=True)
    for k in ("globals", "chosen", "cand_idx", "cand_score", "winner", "margin", "margin5", "turn", "action_mask"):
        np.testing.assert_array_equal(plain[k], aug[k])
    np.testing.assert_array_equal(plain["actions"][..., :2], aug["actions"][..., :2])
    assert not np.array_equal(plain["tiles"], aug["tiles"])
    # per-sample symmetries: each augmented sample is some symmetry of its plain twin
    for i in range(32):
        ok = any(
            np.array_equal(transform_tiles(plain["tiles"][i], s), aug["tiles"][i])
            and np.array_equal(transform_actions(plain["actions"][i], 16, 16, s), aug["actions"][i])
            for s in range(N_SYMMETRIES)
        )
        assert ok, i


def test_loader_buckets_by_map_size(tmp_path):
    paths = []
    for size, seed in ((11, 0), (16, 1), (11, 2)):
        p = tmp_path / f"s{seed}.npz"
        save_shard(p, make_synthetic_shard(40, size, seed))
        paths.append(p)
    loader = BatchLoader(paths, 16, seed=0, prefetch=2)
    sizes = [b["tiles"].shape[1:3] for b in loader.iterate(epochs=2)]
    assert len(sizes) == 2 * 3 * 2  # 40 // 16 = 2 full batches per shard per epoch
    assert set(sizes) == {(11, 11), (16, 16)}
    for b in loader.iterate(epochs=1):
        assert b["tiles"].shape[1] == b["tiles"].shape[2]
        assert b["actions"].shape[0] == 16


def test_loader_without_prefetch_and_partial_batch(tmp_path):
    p = tmp_path / "s.npz"
    save_shard(p, make_synthetic_shard(40, 11, 0))
    loader = BatchLoader([p], 16, augment=False, shuffle=False, drop_last=False, prefetch=0)
    sizes = [b["chosen"].shape[0] for b in loader.iterate(epochs=1)]
    assert sizes == [16, 16, 8]
