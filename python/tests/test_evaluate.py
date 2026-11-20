"""`Game.from_line` and the `evaluate` request (polyai.evaluate, polyai.serve): replay by command text, the protocol,
fog (a seat's number never depends on what it has not seen), nulls with reasons, and batching.

Skipped without the engine module (with `from_line`) or the checkpoint in ai/models.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

pe = pytest.importorskip("polyai_engine")
if not hasattr(pe.Game, "from_line"):
    pytest.skip("the engine module predates Game.from_line", allow_module_level=True)

from polyai.analysis import MODE_NAMES, STAGES, Calibration, ValueNet, save_value  # noqa: E402
from polyai.evaluate import Evaluator  # noqa: E402
from polyai.policy import ModelPolicy  # noqa: E402
from polyai.serve import PolicyClient, PolicyServer  # noqa: E402
from polyai.train_bc import load_checkpoint  # noqa: E402

MODEL = Path(__file__).resolve().parents[2] / "models" / "auto_r0077" / "ckpt_final.pt"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="needs ai/models/auto_r0077")

KINDS = ["human2"] + [f"{m}-{g}" for m in MODE_NAMES.values() for g in ("2p", "3-4p", "5-8p")]


def make_line(seed: int, n_players: int = 2, mode: int = 2, steps: int = 120) -> tuple[str, list[str]]:
    """A start position and the texts of the retail AI's first `steps` commands."""
    g = pe.Game.sample(seed, n_players=[n_players], mode=mode)
    pos = g.position()
    texts = []
    while len(texts) < steps and not g.is_over():
        legal = g.legal_texts()
        i = g.ai_move()
        text = legal[i] if i < len(legal) else "EndTurn"
        g.play_text(text)
        texts.append(text)
    return pos, texts


def value_file(path: Path, kinds=KINDS) -> Path:
    """The live policy's value head with an identity calibration for `kinds` at every stage."""
    net = ValueNet.from_policy(load_checkpoint(MODEL))
    cal = Calibration({k: [(1.0, 0.0, 1.0, 0.0)] * len(STAGES) for k in kinds})
    save_value(path, net, cal, {"name": "test-value"})
    return path


@pytest.fixture(scope="module")
def evaluator():
    with tempfile.TemporaryDirectory() as d:
        policy = ModelPolicy(MODEL, "cpu")
        yield Evaluator(policy, value_file(Path(d) / "v.pt"), "cpu")


@pytest.fixture(scope="module")
def line():
    return make_line(7, n_players=3, mode=2, steps=150)


def steps_of(texts):
    return [{"text": t, "ai": True} for t in texts]


# -- Game.from_line ---------------------------------------------------------------------------------------------------


def test_from_line_replays_to_the_same_state(line):
    pos, texts = line
    g = pe.Game.from_line(pos, texts)
    h = pe.Game.from_position(pos)
    for t in texts:
        h.play_text(t)
    assert g.position() == h.position()
    assert g.turn > 0


def test_from_line_names_the_refused_step(line):
    pos, texts = line
    bad = texts[:5] + ["Move { unit: 9999, to: WorldCoordinates { x: 0, y: 0 } }"] + texts[5:]
    with pytest.raises(ValueError, match=r"^step 5: "):
        pe.Game.from_line(pos, bad)


def test_from_position_refuses_bad_json():
    with pytest.raises(ValueError):
        pe.Game.from_position("{not json")


# -- the request ------------------------------------------------------------------------------------------------------


def test_evaluate_over_the_socket(line):
    pos, texts = line
    with tempfile.TemporaryDirectory() as d:
        policy = ModelPolicy(MODEL, "cpu")
        ev = Evaluator(policy, value_file(Path(d) / "v.pt"), "cpu")
        with PolicyServer(policy, Path(d) / "s.sock", evaluator=ev) as server:
            c = PolicyClient(server.path, timeout=60)
            info = c.info()
            assert "evaluate" in info["ops"] and info["value_id"] == "test-value"
            r = c.evaluate(json.loads(pos), steps_of(texts), at=[0, 10, len(texts)], moves=3, after=True, played=True)
            assert r["ok"], r
            assert r["policy_id"] == policy.policy_id and r["value_id"] == "test-value" and r["engine"] == pe.ENGINE_COMMIT
            assert [e["at"] for e in r["evals"]] == [0, 10, len(texts)]
            for e in r["evals"]:
                assert set(e) >= {"at", "turn", "mover", "win", "why", "moves"}
                assert e["win"], e
                for pid, w in e["win"].items():
                    assert (w is None) == (pid in e["why"])
                    assert w is None or 0.0 <= w <= 1.0
                ps = [m["p"] for m in e["moves"] if not m.get("outside_top")]
                assert ps == sorted(ps, reverse=True) and len(ps) <= 3
                if e["at"] < len(texts):
                    played = [m for m in e["moves"] if m["played"]]
                    assert len(played) == 1 and played[0]["text"] == texts[e["at"]]
                for m in e["moves"]:
                    assert m["win"] is None or 0.0 <= m["win"] <= 1.0
            # defaults: the end of the line, no moves
            r = c.evaluate(json.loads(pos), steps_of(texts))
            assert r["ok"] and [e["at"] for e in r["evals"]] == [len(texts)] and "moves" not in r["evals"][0]
            # bad input never takes the service down
            bad = [
                {"v": 1, "op": "evaluate"},
                {"v": 1, "op": "evaluate", "line": {"position": "{", "steps": []}},
                {"v": 1, "op": "evaluate", "line": {"position": json.loads(pos), "steps": [{"ai": True}]}},
                {"v": 1, "op": "evaluate", "line": {"position": json.loads(pos), "steps": []}, "at": [1]},
                {"v": 1, "op": "evaluate", "line": {"position": json.loads(pos), "steps": []}, "moves": -1},
                {"v": 1, "op": "evaluate", "line": {"position": json.loads(pos), "steps": []}, "seats": "all"},
            ]
            for req in bad:
                out = c.request(req)
                assert out["ok"] is False and out["error"], req
            out = c.evaluate(json.loads(pos), steps_of(texts[:3] + ["Research { tech: 999 }"]), at=[4])
            assert out["ok"] is False and out["error"].startswith("step 3: ")
            assert c.info()["ok"]


def test_moves_need_no_value(line):
    pos, texts = line
    ev = Evaluator(ModelPolicy(MODEL, "cpu"), None, "cpu")
    r = ev.evaluate({"line": {"position": json.loads(pos), "steps": steps_of(texts)}, "at": [0], "moves": 2})
    e = r["evals"][0]
    assert r["value_id"] is None and len(e["moves"]) == 2
    assert all(w is None for w in e["win"].values()) and all("no calibrated value" in x for x in e["why"].values())


# -- fog ----------------------------------------------------------------------------------------------------------------


def hidden_swap(pos: str, pid: int, k: int = 6) -> str:
    """The position with the terrain of `k` pairs of tiles that `pid` has not explored swapped (no city, unit or
    owner on them)."""
    d = json.loads(pos)
    occupied = {(u["x"], u["y"]) for u in d.get("units", []) if "x" in u} | {tuple(u.get("at", ())) for u in d.get("units", [])}
    hidden = [t for t in d["tiles"] if pid not in t.get("explored", []) and t.get("ruling", [-1, -1]) == [-1, -1]
              and (t["x"], t["y"]) not in occupied]
    rng = np.random.default_rng(pid)
    rng.shuffle(hidden)
    changed = 0
    for a, b in zip(hidden[0::2], hidden[1::2]):
        if changed >= k:
            break
        keys = ("terrain", "climate", "improvement", "altitude")
        if all(a.get(x) == b.get(x) for x in keys):
            continue
        for x in keys:
            a[x], b[x] = b.get(x), a.get(x)
        changed += 1
    assert changed > 0, "no hidden tiles to change"
    return json.dumps(d)


@pytest.mark.parametrize("seed", [3, 11])
def test_a_seat_value_ignores_what_it_has_not_seen(evaluator, seed):
    pos, texts = make_line(seed, n_players=3, mode=2, steps=60)
    g = pe.Game.from_line(pos, texts)
    mid = g.position()
    for pid, alive, _ in g.players():
        if not alive:
            continue
        other = hidden_swap(mid, pid)
        assert other != mid
        a, b = pe.Game.from_position(mid), pe.Game.from_position(other)
        ta, ga = a.encode_for(pid)
        tb, gb = b.encode_for(pid)
        assert np.array_equal(ta, tb) and np.array_equal(ga, gb), f"player {pid}'s view changed"
        ra = evaluator.evaluate({"line": {"position": json.loads(mid), "steps": []}, "seats": [pid]})
        rb = evaluator.evaluate({"line": {"position": json.loads(other), "steps": []}, "seats": [pid]})
        assert ra["evals"][0]["win"] == rb["evals"][0]["win"]
        assert ra["evals"][0]["win"][str(pid)] is not None


# -- nulls --------------------------------------------------------------------------------------------------------------


def test_nulls_carry_a_reason(line):
    pos, texts = line
    with tempfile.TemporaryDirectory() as d:
        ev = Evaluator(ModelPolicy(MODEL, "cpu"), value_file(Path(d) / "v.pt", kinds=["human2"]), "cpu")
        r = ev.evaluate({"line": {"position": json.loads(pos), "steps": steps_of(texts)}, "at": [0], "seats": [1, 2, 7], "moves": 1,
                         "after": True})
    e = r["evals"][0]
    assert e["win"]["1"] is None and "domination-3-4p" in e["why"]["1"]
    assert e["win"]["7"] is None and "not a player" in e["why"]["7"]
    assert e["moves"][0]["win"] is None and e["moves"][0]["why"]


def test_an_unsupported_position_gives_nulls_not_an_error(evaluator, line):
    pos, texts = line
    d = json.loads(pos)
    d["players"][0]["tribe"] = 99
    r = evaluator.evaluate({"line": {"position": d, "steps": []}, "at": [0], "moves": 3})
    for e in r["evals"]:
        assert e["win"] and all(w is None for w in e["win"].values())
        assert all("tribe" in why for why in e["why"].values())
        assert e["moves"] == []


def test_finished_and_eliminated_players_are_exact(evaluator):
    pos, texts = make_line(5, n_players=2, mode=2, steps=20000)
    g = pe.Game.from_line(pos, texts)
    if not g.is_over():
        pytest.skip("the sampled game did not finish")
    r = evaluator.evaluate({"line": {"position": json.loads(pos), "steps": steps_of(texts)}, "seats": [1, 2]})
    w = r["evals"][0]["win"]
    assert sorted(w.values()) == [0.0, 1.0] and w[str(g.winner_ids()[0])] == 1.0


# -- batching -----------------------------------------------------------------------------------------------------------


def test_one_request_for_many_points_equals_one_by_one(evaluator, line):
    pos, texts = line
    at = list(range(0, len(texts) + 1, 37))
    base = {"line": {"position": json.loads(pos), "steps": steps_of(texts)}, "moves": 2, "after": True, "played": True}
    many = evaluator.evaluate(base | {"at": at})["evals"]
    for i, e in zip(at, many):
        one = evaluator.evaluate(base | {"at": [i]})["evals"][0]
        assert (one["turn"], one["mover"], one["why"]) == (e["turn"], e["mover"], e["why"])
        assert one["win"].keys() == e["win"].keys()
        for k in one["win"]:
            assert one["win"][k] is None and e["win"][k] is None or abs(one["win"][k] - e["win"][k]) < 2e-3
        assert [m["text"] for m in one["moves"]] == [m["text"] for m in e["moves"]]
        for a, b in zip(one["moves"], e["moves"]):
            assert abs(a["p"] - b["p"]) < 2e-3
            assert (a["win"] is None) == (b["win"] is None)
            assert a["win"] is None or abs(a["win"] - b["win"]) < 2e-3
