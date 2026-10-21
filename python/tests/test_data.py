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
    assert S.G_DIM == 13 + 12 + 51 + 7 * 24 + 3 == 247
    assert S.G_SLOTS + 7 * S.PER_SLOT == S.G_TRIGGER


def test_synthetic_shard_matches_schema():
    arrays = make_synthetic_shard(32, 11, seed=3)
    shard = Shard(arrays)
    assert (shard.n, shard.h, shard.w) == (32, 11, 11)
    for name, spec in S.SHARD_FIELDS.items():
        assert arrays[name].dtype == np.dtype(spec.dtype), name
    for i, v in enumerate(S.CAT_VOCABS):
        assert arrays["tiles"][..., i].max() < v


def v1_of(arrays: dict) -> dict:
    """A synthetic shard in the version 1 layout: top-score adjudication, margins, a turn limit in the globals."""
    out = {k: v for k, v in arrays.items() if k != "capped"}
    n = len(arrays["winner"])
    adjudicated = np.zeros(n, np.int8)
    adjudicated[::5] = 1
    winner = out["winner"].copy()
    winner[adjudicated == 1] = 0  # v1 gave an adjudicated game a winner: the top scorer
    winner[arrays["capped"] == 1] = 1
    out["winner"] = winner
    out["adjudicated"] = adjudicated
    out["margin"] = np.linspace(-9000, 9000, n).astype(np.float32)
    out["margin5"] = out["margin"] * 0.5
    out["globals"] = out["globals"].copy()
    out["globals"][:, S.G_TURNS_LEFT] = 0.6
    out["globals"][:, S.G_HAS_LIMIT] = 1.0
    out["version"] = np.array([1], np.int32)
    return out


def test_v2_shard_roundtrip_keeps_everything(tmp_path):
    arrays = make_synthetic_shard(40, 11, seed=8)
    assert int(arrays["version"][0]) == 2 and "margin" not in arrays and "adjudicated" not in arrays
    assert (arrays["capped"] == 1).any() and (arrays["winner"][arrays["capped"] == 1] == -1).all()
    save_shard(tmp_path / "s.npz", arrays)
    shard = Shard.load(tmp_path / "s.npz")
    assert shard.source_version == 2
    for k in S.REQUIRED_FIELDS:
        np.testing.assert_array_equal(shard.a[k], arrays[k])


def test_v1_shard_is_relabelled_on_load(tmp_path):
    old = v1_of(make_synthetic_shard(40, 11, seed=8))
    np.savez(tmp_path / "old.npz", **old)
    shard = Shard.load(tmp_path / "old.npz")
    assert shard.source_version == 1 and (shard.n, shard.h) == (40, 11)
    adj = old["adjudicated"] == 1
    assert adj.any()
    assert (shard.a["winner"][adj] == -1).all()  # adjudicated games are draws now
    np.testing.assert_array_equal(shard.a["winner"][~adj], old["winner"][~adj])
    np.testing.assert_array_equal(shard.a["capped"], adj.astype(np.int8))
    assert (shard.a["globals"][:, [S.G_TURNS_LEFT, S.G_HAS_LIMIT]] == 0).all()  # no turn-limit information
    assert old["globals"][0, S.G_TURNS_LEFT] == pytest.approx(0.6)  # the input was not touched
    assert "margin" not in shard.a and "margin5" not in shard.a and "adjudicated" not in shard.a
    b = shard.batch(np.arange(10))
    assert "margin" not in b and (b["winner"][adj[:10]] == -1).all()


def test_v1_without_its_own_fields_and_unknown_versions_are_rejected():
    old = v1_of(make_synthetic_shard(8, 11))
    del old["adjudicated"]
    with pytest.raises(ValueError, match="adjudicated"):
        Shard(old)
    bad = make_synthetic_shard(8, 11)
    bad["version"] = np.array([3], np.int32)
    with pytest.raises(ValueError, match="version 3"):
        Shard(bad)


def test_a_capped_game_with_a_winner_is_rejected():
    arrays = make_synthetic_shard(8, 11)
    arrays["capped"][:] = 1
    arrays["winner"][:] = 0
    with pytest.raises(ValueError, match="capped"):
        Shard(arrays)


def test_save_and_load_roundtrip(tmp_path):
    arrays = make_synthetic_shard(16, 11, seed=4)
    save_shard(tmp_path / "s.npz", arrays)
    loaded = Shard.load(tmp_path / "s.npz")
    for k in S.REQUIRED_FIELDS:
        np.testing.assert_array_equal(loaded.a[k], arrays[k])


def test_missing_field_is_rejected(tmp_path):
    arrays = make_synthetic_shard(4, 11)
    del arrays["capped"]
    with pytest.raises(ValueError, match="capped"):
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
    for k in ("globals", "chosen", "cand_idx", "cand_score", "winner", "turn", "action_mask"):
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
