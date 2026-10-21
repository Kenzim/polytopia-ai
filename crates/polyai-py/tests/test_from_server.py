"""Game.from_server, raw_command, index_of_raw and describe, on real server state blobs (docs/interfaces.md 1.2).

The blobs are the `*_init.bin` / `*_cur.bin` pairs in `re/data/regress` (read only); the tests are skipped when that
directory is absent.
"""
import struct
from pathlib import Path

import numpy as np
import pytest

import polyai_engine as pe

REGRESS = Path("/kvm/polytopia/re/data/regress")
PAIRS = sorted(REGRESS.glob("*_init.bin")) if REGRESS.is_dir() else []
needs_blobs = pytest.mark.skipif(not PAIRS, reason="no server blobs in re/data/regress")


def blobs(init: Path) -> tuple[bytes, bytes]:
    return init.read_bytes(), init.with_name(init.name.replace("_init.bin", "_cur.bin")).read_bytes()


def rebuild(init: Path):
    """The game and the seat it was built for, or the reason it cannot be used (a ValueError message)."""
    i, c = blobs(init)
    for seat in range(0, 12):
        try:
            return pe.Game.from_server(i, c, seat), seat
        except ValueError as e:
            if "turn, not seat" not in str(e) and "not a player" not in str(e):
                return str(e), None
    return "no seat is to move", None


def usable():
    out = []
    for p in PAIRS:
        g, seat = rebuild(p)
        if seat is not None:
            out.append((p, g, seat))
    return out


@needs_blobs
def test_regress_blobs_rebuild_encode_and_round_trip():
    games = usable()
    assert len(games) >= 8
    rejected = {p.name: rebuild(p)[0] for p in PAIRS if rebuild(p)[1] is None}
    # The ones that are refused are refused for a stated reason, not by accident.
    assert all(("players; the encoding holds" in r) or ("not one of the 13" in r) for r in rejected.values()), rejected
    total = 0
    for p, g, seat in games:
        assert g.mover_id == seat and not g.is_over()
        tiles, glob = g.encode()
        h, w = g.shape
        assert tiles.shape == (h, w, pe.F_TILE) and glob.shape == (pe.G_DIM,)
        acts = g.legal_actions()
        n = len(acts)
        assert n > 0 and acts.shape == (n, 4)
        assert (acts[:, 0] == 26).sum() >= 1  # EndTurn is there
        seen = set()
        for i in range(n):
            raw = g.raw_command(i)
            assert isinstance(raw, bytes) and len(raw) >= 3
            assert g.index_of_raw(raw) == i, (p.name, i, g.describe(i))
            assert isinstance(g.describe(i), str) and g.describe(i)
            seen.add(raw)
        assert len(seen) == n  # two legal entries never share bytes
        total += n
        # Raw commands carry the mover's id.
        assert all(g.raw_command(i)[2] == seat for i in range(n))
    assert total > 1000


@needs_blobs
def test_server_game_can_be_played_on():
    p, g, seat = usable()[0]
    n0 = len(g.legal_actions())
    end = [i for i in range(n0) if g.legal_actions()[i][0] == 26][0]
    g.step(end)  # ends the turn; the engine moves on to the next player
    assert g.mover_id != seat or g.turn > 0
    with pytest.raises(ValueError):
        g.setup()
    with pytest.raises(ValueError):
        g.record()


@needs_blobs
def test_the_current_state_is_not_modified_by_the_clone():
    p, g, seat = usable()[0]
    c = g.clone()
    c.step(0)
    assert len(g.legal_actions()) == len(pe.Game.from_server(*blobs(p), seat).legal_actions())


@needs_blobs
def test_bad_input_is_a_value_error():
    p, g, seat = usable()[0]
    i, c = blobs(p)
    for bad in [(b"", b"", seat), (b"junk", b"junk", seat), (i, c[:200], seat), (i[:100], c, seat), (c, i, seat)]:
        with pytest.raises(ValueError):
            pe.Game.from_server(*bad)
    other = 1 if seat != 1 else 2  # every game here has players 1 and 2
    with pytest.raises(ValueError, match="turn, not seat"):
        pe.Game.from_server(i, c, other)
    with pytest.raises(ValueError, match="not a player"):
        pe.Game.from_server(i, c, 99)
    # Flipped bytes inside the state: an error or a (different) game, never a crash of the interpreter.
    rng = np.random.default_rng(0)
    for _ in range(20):
        b = bytearray(c)
        for k in rng.integers(4, len(b), size=8):
            b[k] ^= 0xFF
        try:
            pe.Game.from_server(i, bytes(b), seat)
        except ValueError:
            pass


@needs_blobs
def test_too_many_players_and_unsupported_tribe_are_refused():
    names = {p.name: rebuild(p)[0] for p in PAIRS}
    assert any(isinstance(v, str) and "players; the encoding holds at most 8" in v for v in names.values())
    assert any(isinstance(v, str) and "not one of the 13 supported tribes" in v for v in names.values())


def test_raw_command_round_trips_on_generated_games():
    """Every legal entry (Stay included) of positions along real games maps to bytes and back to the same index."""
    checked = stays = 0
    for seed in (3, 11):
        g = pe.Game.sample(seed, n_players=[2, 3], sizes=[1], turn_limit=40)
        while not g.is_over():
            if g.current_seat() is not None and checked % 7 == 0:
                acts = g.legal_actions()
                for i in range(len(acts)):
                    assert g.index_of_raw(g.raw_command(i)) == i, g.describe(i)
                    stays += acts[i][0] == 10
            checked += 1
            g.play_ai_move()
    assert stays > 20


def test_raw_command_errors():
    g = pe.Game.sample(3, n_players=[2], sizes=[1])
    n = len(g.legal_actions())
    with pytest.raises(IndexError):
        g.raw_command(n)
    with pytest.raises(IndexError):
        g.describe(n)
    with pytest.raises(ValueError):
        g.index_of_raw(b"")
    with pytest.raises(ValueError):
        g.index_of_raw(b"\xff\xff\x01")
    # A well-formed command that is not in the legal list: destroying a building on a tile that has none.
    with pytest.raises(ValueError, match="not in the legal list"):
        g.index_of_raw(struct.pack("<HBii", 9, 1, 0, 0))
