"""The model arena: collation, agent names, and a short end-to-end run on the real engine module (skips without it)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from polyai import schema as S
from polyai.arena import (
    ModelAgent,
    Suite,
    agent_label,
    collate,
    game_record,
    load_done,
    parse_model_spec,
    parse_opponent,
    play_pairing,
    summarise,
)
from polyai.model import build_model
from polyai.train_bc import save_checkpoint

ELO = Path(__file__).resolve().parents[2] / "tools" / "elo.py"


def fake_obs(lens=(3, 5, 2), size=11, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    b = len(lens)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    actions = np.stack([rng.integers(1, 26, off[-1]), rng.integers(0, 5, off[-1]),
                        rng.integers(0, size * size, off[-1]), rng.integers(0, size * size, off[-1])], 1).astype(np.int16)
    tiles = rng.integers(0, 3, (b, size, size, S.F_TILE)).astype(np.int16)
    return {"game": np.arange(b, dtype=np.int32), "seat": np.zeros(b, np.int8), "n_players": np.full(b, 2, np.int8),
            "turn": np.ones(b, np.int16), "seed": np.arange(b, dtype=np.int32) + 1_000_000, "tiles": tiles,
            "globals": rng.normal(size=(b, S.G_DIM)).astype(np.float32), "act_off": off, "actions": actions}


def test_collate_pads_each_legal_list_and_selects_positions():
    obs = fake_obs()
    tiles, globals_, actions, mask = collate(obs, np.array([2, 0]), torch.device("cpu"))
    assert tiles.shape[0] == 2 and globals_.shape == (2, S.G_DIM) and actions.shape == (2, 3, 4)
    assert mask.tolist() == [[True, True, False], [True, True, True]]
    np.testing.assert_array_equal(actions[0, :2].numpy(), obs["actions"][8:10])  # position 2 owns rows 8 and 9
    np.testing.assert_array_equal(actions[1].numpy(), obs["actions"][0:3])
    assert (actions[0, 2].numpy() == [S.KIND_PAD, 0, -1, -1]).all()


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    run = tmp_path_factory.mktemp("runs") / "tinyrun"
    run.mkdir()
    torch.manual_seed(0)
    model = build_model("tiny")
    for p in model.parameters():  # non-zero heads, so that greedy and sampled picks differ
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.3)
    save_checkpoint(run / "ckpt_final.pt", model, 7)
    return run / "ckpt_final.pt"


def test_model_specs_and_names(checkpoint):
    run = checkpoint.parent
    assert parse_model_spec(str(run)) == (checkpoint, None)
    assert parse_model_spec(f"model:{checkpoint}") == (checkpoint, None)
    assert parse_model_spec(f"{checkpoint}:sample=0.5") == (checkpoint, 0.5)
    assert agent_label(checkpoint, None) == "model:tinyrun"
    assert agent_label(checkpoint, 0.5) == "model:tinyrun:sample=0.5"
    assert agent_label(run / "ckpt_last.pt", None) == "model:tinyrun/ckpt_last"
    with pytest.raises(FileNotFoundError):
        parse_model_spec(str(run / "missing.pt"))
    with pytest.raises(ValueError):
        parse_model_spec(f"{checkpoint}:sample=0")
    assert [parse_opponent(f"retail:{d}") for d in ("easy", "normal", "hard", "crazy")] == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        parse_opponent("random")


def test_agent_picks_legal_indices_greedy_and_sampled(checkpoint):
    obs = fake_obs(lens=(3, 5, 2, 9))
    lens = np.diff(obs["act_off"])
    greedy = ModelAgent(checkpoint, None, torch.device("cpu"))
    a = greedy.choose(obs)
    assert a.dtype == np.int32 and (a >= 0).all() and (a < lens).all()
    np.testing.assert_array_equal(a, greedy.choose(obs))  # deterministic
    assert greedy.step == 7 and greedy.decisions == 8
    sampled = ModelAgent(checkpoint, 1.0, torch.device("cpu"), seed=1)
    picks = np.stack([sampled.choose(obs) for _ in range(30)])
    assert (picks < lens).all() and len(set(map(tuple, picks))) > 1
    small = ModelAgent(checkpoint, 1.0, torch.device("cpu"), max_batch=2, seed=1)  # chunking gives legal picks too
    assert (small.choose(obs) < lens).all()


def test_capped_games_are_draws_whatever_the_engine_says():
    pe = pytest.importorskip("polyai_engine")
    setup = pe.sample_setup(1_000_000, n_players=[2], turn_limit=150)
    f = {"seed": 1_000_000, "tribes": [2, 4], "winners": [1], "scores": [10, 20], "turns": 150, "commands": 99,
         "difficulties": [1, 0], "adjudicated": True}
    r = game_record(f, pe, Suite(), 0, ["model:x", "retail:easy"], setup, "x.pt")
    assert r["winner"] is None and r["adjudicated"] and r["capped"]
    f2 = {**f, "adjudicated": False, "capped": False}
    assert game_record(f2, pe, Suite(), 0, ["model:x", "retail:easy"], setup, "x.pt")["winner"] == 1
    assert game_record({**f2, "winners": []}, pe, Suite(), 0, ["a", "b"], setup, "x.pt")["winner"] is None


def test_summary_counts_from_the_models_seat():
    rec = lambda agents, w, capped=False: {"agents": agents, "winner": w, "capped": capped, "turns": 10}  # noqa: E731
    rs = [rec(["m", "r"], 0), rec(["r", "m"], 0), rec(["r", "m"], 1), rec(["m", "r"], None, True), rec(["x", "y"], 0)]
    s = summarise(rs, "m", "r")
    assert (s["wins"], s["losses"], s["draws"], s["capped"], s["games"]) == (2, 1, 1, 1, 4)
    assert s["score"] == pytest.approx(2.5 / 4)


def test_end_to_end_on_the_engine_and_elo_reads_the_file(checkpoint, tmp_path):
    pe = pytest.importorskip("polyai_engine")
    suite = Suite(seeds=3, sizes=[1], presets=[3], turn_limit=12, batch_games=2, threads=4)
    agent = ModelAgent(checkpoint, None, torch.device("cpu"))
    records = play_pairing(agent, "retail:easy", suite, log=lambda m: None)
    assert len(records) == 6
    assert {(r["seed"], r["swap"]) for r in records} == {(s, w) for s in range(suite.seed_start, suite.seed_start + 3) for w in (0, 1)}
    for r in records:
        assert r["agents"] == (["model:tinyrun", "retail:easy"] if r["swap"] == 0 else ["retail:easy", "model:tinyrun"])
        assert r["setup"]["size"] == 1 and r["setup"]["preset"] == 3 and r["setup"]["mode"] == 2
        assert r["winner"] in (None, 0, 1) and len(r["scores"]) == 2 and len(r["tribes"]) == 2
        assert r["capped"] == r["adjudicated"] and (r["winner"] is None or not r["capped"])
        assert r["turns"] <= 13 and r["over"] and r["illegal"] == [0, 0]
    # same seed, other seating: same tribes per seat and the same map
    by = {(r["seed"], r["swap"]): r for r in records}
    for s in range(suite.seed_start, suite.seed_start + 3):
        assert by[(s, 0)]["tribes"] == by[(s, 1)]["tribes"] and by[(s, 0)]["setup"] == by[(s, 1)]["setup"]
    out = tmp_path / "a.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert len(load_done(out)) == 6
    assert play_pairing(agent, "retail:easy", suite, done=load_done(out), log=lambda m: None) == []
    spec = importlib.util.spec_from_file_location("elo", ELO)
    elo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(elo)
    games = elo.load_games([out])
    assert len(games) == 6
    stats = elo.pair_stats(games)
    assert sum(s["n"] for s in stats.values()) == 6


def test_a_three_player_suite_seats_the_model_once_in_every_seat(checkpoint):
    pytest.importorskip("polyai_engine")
    suite = Suite(seeds=2, sizes=[2], presets=[1], turn_limit=12, batch_games=2, threads=4, n_players=3)
    agent = ModelAgent(checkpoint, None, torch.device("cpu"))
    records = play_pairing(agent, "retail:easy", suite, log=lambda m: None)
    assert {(r["seed"], r["swap"]) for r in records} == {(s, w) for s in (1_000_000, 1_000_001) for w in range(3)}
    for r in records:
        assert r["agents"] == ["model:tinyrun" if s == r["swap"] else "retail:easy" for s in range(3)]
        assert len(r["tribes"]) == len(r["scores"]) == len(r["think_ms"]) == len(r["illegal"]) == 3
        assert r["seat_difficulties"] == [1 if s == r["swap"] else 0 for s in range(3)]
        assert r["winner"] in (None, 0, 1, 2)
    s = summarise(records, "model:tinyrun", "retail:easy")
    assert s["games"] == 6 and s["wins"] + s["losses"] + s["draws"] == 6


def test_per_seat_difficulties_are_used_when_the_engine_offers_them():
    """The model seat plays at Normal handicap, as in polyai-arena, once VecEnv accepts `seat_difficulties`."""
    seen = []

    class FakeEnv:
        """Stands for the pyo3 class; its signature is what the runner inspects."""

        def __init__(self, signature):
            self.__text_signature__ = signature

        def __call__(self, n_games, seed_start, model_seats, **kw):
            seen.append((n_games, seed_start, model_seats, kw))
            return self

        def observe(self):
            return []

    class FakePe:
        VecEnv = FakeEnv("(n_games, seed_start, model_seats, seat_difficulties=None, difficulty=None)")

    from polyai.arena import Arm

    Arm(FakePe, Suite(seeds=5, batch_games=3), 3, 1, 2)
    n, start, seats, kw = seen[0]
    assert (n, start, seats) == (3, 1_000_000, [1])
    assert kw["seat_difficulties"] == [3, 1] and "difficulty" not in kw
    assert kw["n_players"] == [2] and kw["split"] == "eval" and kw["turn_limit"] == 150

    FakePe.VecEnv = FakeEnv("(n_games, seed_start, model_seats, difficulty=None)")
    Arm(FakePe, Suite(seeds=5), 2, 0, 2)
    assert seen[1][3]["difficulty"] == 2 and "seat_difficulties" not in seen[1][3]

    FakePe.VecEnv = FakeEnv("(n_games, seed_start, model_seats, seat_difficulties=None, difficulty=None)")
    Arm(FakePe, Suite(seeds=5, n_players=4), 3, 2, 2)
    assert seen[2][2] == [2] and seen[2][3]["n_players"] == [4] and seen[2][3]["seat_difficulties"] == [3, 3, 1, 3]
