"""`polyai.serve` end to end on CPU: a real checkpoint, real server state blobs, the socket protocol, bad input.

Skipped without the engine module, the checkpoint run (`bc3_b6c96_v11`) or the blobs in `re/data/regress`.
"""

import base64
import json
import os
import shutil
import socket
import stat
import struct
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

pe = pytest.importorskip("polyai_engine")

from polyai.model import eval_pawns  # noqa: E402
from polyai.policy import ModelPolicy, observe_games  # noqa: E402
from polyai.serve import PolicyClient, PolicyServer  # noqa: E402

RUN = Path("/kvm/polytopia-ai-data/runs/bc3_b6c96_v11")
REGRESS = Path("/kvm/polytopia/re/data/regress")
PAIRS = sorted(REGRESS.glob("*_init.bin")) if REGRESS.is_dir() else []
pytestmark = pytest.mark.skipif(not (RUN / "ckpt_final.pt").exists() or not PAIRS, reason="needs the bc3 run and the regress blobs")


def blobs(init: Path) -> tuple[bytes, bytes]:
    return init.read_bytes(), init.with_name(init.name.replace("_init.bin", "_cur.bin")).read_bytes()


def find_seat(i: bytes, c: bytes):
    for seat in range(0, 12):
        try:
            return pe.Game.from_server(i, c, seat), seat, None
        except ValueError as e:
            if "turn, not seat" not in str(e) and "not a player" not in str(e):
                return None, None, str(e)
    return None, None, "no seat"


@pytest.fixture(scope="module")
def positions():
    out, refused = [], {}
    for p in PAIRS:
        i, c = blobs(p)
        g, seat, why = find_seat(i, c)
        if g is None:
            refused[p.name] = why
        else:
            out.append((p.name[:-9], i, c, seat))
    assert len(out) >= 8 and refused
    return out, refused


@pytest.fixture(scope="module")
def policy():
    return ModelPolicy(RUN, "cpu")


@pytest.fixture()
def server(policy):
    d = tempfile.mkdtemp(prefix="ps", dir=os.environ.get("TMPDIR", "/tmp"))
    srv = PolicyServer(policy, Path(d) / "p.sock", window_ms=3.0).start()
    yield srv
    srv.close()
    shutil.rmtree(d, ignore_errors=True)


def test_socket_mode_and_info(server, policy):
    assert stat.S_IMODE(os.stat(server.path).st_mode) == 0o600
    with PolicyClient(server.path) as c:
        r = c.info()
        assert r["ok"] and r["v"] == 1 and r["policy_id"] == "bc3_b6c96_v11@40000" == policy.policy_id
        assert r["encoding"] == pe.ENCODING_VERSION and r["engine"] == pe.ENGINE_COMMIT


def test_choose_on_real_states_matches_the_policy_and_maps_back(server, policy, positions):
    ps, _ = positions
    with PolicyClient(server.path) as c:
        for name, i, c_blob, seat in ps:
            r = c.choose(name, seat, i, c_blob)
            assert r["ok"], (name, r)
            game = pe.Game.from_server(i, c_blob, seat)
            direct = policy.choose(observe_games([game]))
            n = len(game.legal_actions())
            assert r["n_legal"] == n and r["index"] == int(direct.index[0]) and 0 <= r["index"] < n
            raw = base64.b64decode(r["raw"])
            assert raw == game.raw_command(r["index"]) and game.index_of_raw(raw) == r["index"]  # the client's check
            assert r["desc"] == game.describe(r["index"])
            assert r["logp"] == pytest.approx(float(direct.logp[0]), abs=1e-4) and r["logp"] <= 0
            assert r["value"] == pytest.approx(float(direct.value[0]), abs=1e-5) and 0 < r["value"] < 1
            assert r["eval"] == pytest.approx(float(eval_pawns(r["value"])), abs=1e-4)
            assert len(r["top"]) == min(3, n) and r["top"][0]["index"] == r["index"]
            ps_ = [t["p"] for t in r["top"]]
            assert ps_ == sorted(ps_, reverse=True) and r["top"][0]["desc"] == r["desc"]
            assert r["policy_id"] == policy.policy_id and r["encoding"] == pe.ENCODING_VERSION
            assert r["engine"] == pe.ENGINE_COMMIT and r["think_ms"] > 0 and "forced" not in r


def test_reply_applies_to_the_local_game(server, positions):
    """What a client does with a reply: map the raw bytes back to a legal entry and apply it to the rebuilt game."""
    ps, _ = positions
    for name, i, c_blob, seat in ps[:4]:
        game = pe.Game.from_server(i, c_blob, seat)
        with PolicyClient(server.path) as c:
            r = c.choose(name, seat, i, c_blob)
        game.step(game.index_of_raw(base64.b64decode(r["raw"])))
        assert game.mover_id is not None


def test_sampling_temperature(server, positions):
    ps, _ = positions
    name, i, c_blob, seat = ps[0]
    with PolicyClient(server.path) as c:
        greedy = c.choose(name, seat, i, c_blob)
        picks = {c.choose(name, seat, i, c_blob, temperature=3.0)["index"] for _ in range(25)}
        assert len(picks) > 1 and all(0 <= p < greedy["n_legal"] for p in picks)
        assert c.choose(name, seat, i, c_blob, temperature=1)["ok"]  # an int is a number


def test_turn_ends_after_400_commands(server, positions):
    ps, _ = positions
    name, i, c_blob, seat = ps[0]
    game = pe.Game.from_server(i, c_blob, seat)
    with PolicyClient(server.path) as c:
        r = c.choose(name, seat, i, c_blob, per_turn=399)
        assert r["ok"] and "forced" not in r
        r = c.choose(name, seat, i, c_blob, per_turn=400)
        assert r["ok"] and game.legal_actions()[r["index"]][0] == 26 and r["desc"] == "EndTurn"
        assert "forced" in r or r["index"] == c.choose(name, seat, i, c_blob)["index"]


def test_refused_states_get_a_clear_error(server, positions):
    ps, refused = positions
    with PolicyClient(server.path) as c:
        for name, why in refused.items():
            i, cur = blobs(REGRESS / name)
            r = c.choose(name, 1, i, cur)
            assert not r["ok"] and r["v"] == 1 and r["error"]
        name, i, cur, seat = ps[0]
        other = 1 if seat != 1 else 2
        r = c.choose(name, other, i, cur)
        assert not r["ok"] and "not seat" in r["error"]
        r = c.choose(name, 99, i, cur)
        assert not r["ok"] and "not a player" in r["error"]
        assert c.choose(name, seat, i, cur)["ok"]  # the same connection still works


def test_bad_requests_never_crash_the_service(server, positions):
    ps, _ = positions
    name, i, cur, seat = ps[0]
    good = {"v": 1, "op": "choose", "game_id": "g", "seat": seat, "initial": base64.b64encode(i).decode(),
            "current": base64.b64encode(cur).decode(), "per_turn": 0, "temperature": None}

    def with_(**kw):
        return {**good, **kw}

    def without(k):
        return {a: b for a, b in good.items() if a != k}

    bad = [
        with_(v=2), with_(v=True), with_(v="1"), without("v"), with_(op="move"), with_(op=None), without("op"),
        without("seat"), with_(seat="2"), with_(seat=-1), with_(seat=256), with_(seat=1.5), with_(seat=True),
        without("initial"), without("current"), with_(initial=5), with_(current="@@@@"), with_(initial=""),
        with_(initial=base64.b64encode(b"junk").decode()), with_(current=base64.b64encode(cur[:300]).decode()),
        with_(current=base64.b64encode(i).decode(), initial=base64.b64encode(cur).decode()),
        with_(per_turn=-1), with_(per_turn="x"), with_(per_turn=1.5),
        with_(temperature=0), with_(temperature=-1.0), with_(temperature="hot"), with_(temperature=[1]),
        with_(temperature=float("nan")), with_(temperature=float("inf")), with_(game_id=7),
    ]
    with PolicyClient(server.path) as c:
        for req in bad:
            r = c.request(req)
            assert r["ok"] is False and r["v"] == 1 and isinstance(r["error"], str) and r["error"], req
        for payload in [b"", b"not json", b"\xff\xfe\x00", b"[]", b"7", b"null", b'"x"', b"{", b"{}"]:
            r = c.request_raw(payload)
            assert r["ok"] is False and r["error"], payload
        assert c.choose(name, seat, i, cur)["ok"]  # all of that on one connection, which still works
    # Damaged blobs: an error or an answer, never a dead service.
    rng = np.random.default_rng(1)
    with PolicyClient(server.path) as c:
        answers = set()
        for _ in range(25):
            b = bytearray(cur)
            for k in rng.integers(4, len(b), size=6):
                b[k] ^= 0xFF
            r = c.choose(name, seat, i, bytes(b))
            assert isinstance(r["ok"], bool)
            answers.add(r["ok"])
        assert c.info()["ok"]


def test_oversized_and_truncated_messages(server, positions):
    # A header announcing too much is answered once, then the connection is closed (the stream cannot be resynchronised).
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(5)
    s.connect(str(server.path))
    s.sendall(struct.pack("<I", 0xFFFFFFF0))
    head = s.recv(4)
    r = json.loads(s.recv(struct.unpack("<I", head)[0]))
    assert r["ok"] is False and "limit" in r["error"]
    assert s.recv(1) == b""
    s.close()
    # A client that goes away half way through a message does not hurt the service.
    s = socket.socket(socket.AF_UNIX)
    s.connect(str(server.path))
    s.sendall(struct.pack("<I", 1000) + b"{")
    s.close()
    with PolicyClient(server.path) as c:
        assert c.info()["ok"]


def test_concurrent_requests_are_batched_and_answered_correctly(policy, positions):
    ps, _ = positions
    d = tempfile.mkdtemp(prefix="ps", dir=os.environ.get("TMPDIR", "/tmp"))
    with PolicyServer(policy, Path(d) / "p.sock", window_ms=200.0) as srv:
        results, errors = {}, []
        start = threading.Barrier(len(ps))

        def work(k):
            name, i, cur, seat = ps[k]
            try:
                with PolicyClient(srv.path) as c:
                    start.wait()
                    results[k] = c.choose(name, seat, i, cur)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=work, args=(k,)) for k in range(len(ps))]
        [t.start() for t in threads]
        [t.join(60) for t in threads]
        assert not errors and len(results) == len(ps)
        assert srv.batcher.largest_batch >= 2
        for k, (name, i, cur, seat) in enumerate(ps):
            direct = policy.choose(observe_games([pe.Game.from_server(i, cur, seat)]))
            r = results[k]
            assert r["ok"] and r["index"] == int(direct.index[0])
            assert r["logp"] == pytest.approx(float(direct.logp[0]), abs=1e-3)
            assert r["value"] == pytest.approx(float(direct.value[0]), abs=1e-3)
    shutil.rmtree(d, ignore_errors=True)


def test_stale_socket_is_replaced_and_a_live_one_is_not(policy):
    d = tempfile.mkdtemp(prefix="ps", dir=os.environ.get("TMPDIR", "/tmp"))
    path = Path(d) / "p.sock"
    dead = socket.socket(socket.AF_UNIX)
    dead.bind(str(path))
    dead.close()  # leaves the file, nobody listens
    with PolicyServer(policy, path) as srv:
        with PolicyClient(srv.path) as c:
            assert c.info()["ok"]
        with pytest.raises(RuntimeError, match="already listening"):
            PolicyServer(policy, path).bind()
    assert not path.exists()
    shutil.rmtree(d, ignore_errors=True)
