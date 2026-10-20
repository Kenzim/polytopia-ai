"""Smoke tests for the polyai_engine module (run after `maturin develop`)."""
import time

import numpy as np
import pytest

import polyai_engine as pe


def test_constants():
    assert pe.ENCODING_VERSION == 1
    assert pe.F_TILE == 41 and pe.G_DIM == 247 and pe.N_KINDS == 32
    assert len(pe.TRIBES) == 13


def test_game_ai_play_and_record():
    g = pe.Game.sample(5, n_players=[2], sizes=[1], turn_limit=12)
    assert g.shape == (11, 11)
    n = 0
    while not g.is_over():
        seat = g.current_seat()
        if seat is not None:
            tiles, glob = g.encode()
            acts = g.legal_actions()
            assert tiles.shape == (11, 11, 41) and tiles.dtype == np.int16
            assert glob.shape == (247,) and glob.dtype == np.float32
            assert acts.shape[1] == 4 and acts.dtype == np.int16
            idx = g.ai_move()
            assert 0 <= idx < len(acts)
            if n % 40 == 0:
                cand, score = g.ai_candidates()
                assert len(cand) == len(score) <= 16
        g.play_ai_move()
        n += 1
    assert g.winners()
    assert len(g.scores()) == 2
    assert g.record()[:4] == b"PREC"


def test_step_action_matches_step():
    a = pe.Game.sample(9, n_players=[2], sizes=[1])
    b = a.clone()
    for _ in range(30):
        if a.current_seat() is None:
            a.play_ai_move()
            b.play_ai_move()
            continue
        acts = a.legal_actions()
        i = len(acts) // 2
        a.step(i)
        b.step_action(tuple(int(x) for x in acts[i]))
    assert a.scores() == b.scores() and a.turn == b.turn
    with pytest.raises(ValueError):
        b.step_action((99, 0, 0, 0))


def test_setup_round_trip():
    s = pe.sample_setup(77)
    g = pe.Game.from_setup(s)
    assert g.setup() == s


def test_seed_split_is_enforced():
    with pytest.raises(ValueError):
        pe.VecEnv(2, seed_start=5, split="eval")
    with pytest.raises(ValueError):
        pe.VecEnv(2, seed_start=2_000_000, split="train")


def check_groups(groups, env_n):
    seen = 0
    for g in groups:
        b = len(g["game"])
        seen += b
        assert g["tiles"].shape[0] == b and g["tiles"].shape[3] == 41
        assert g["globals"].shape == (b, 247)
        assert g["act_off"].shape == (b + 1,) and g["act_off"][0] == 0
        assert g["actions"].shape == (g["act_off"][-1], 4)
        assert (np.diff(g["act_off"]) > 0).all()
    return seen


@pytest.mark.parametrize("mode", ["rotate", "all", [0]])
def test_vecenv_random_policy(mode):
    env = pe.VecEnv(16, seed_start=1, model_seats=mode, sizes=[1], turn_limit=15, threads=4)
    rng = np.random.default_rng(0)
    groups = env.observe()
    assert check_groups(groups, 16) == 16
    finished_total = 0
    decisions = 0
    for _ in range(400):
        ids, choices = [], []
        for g in groups:
            lens = np.diff(g["act_off"])
            ids.append(g["game"])
            choices.append((rng.random(len(lens)) * lens).astype(np.int32))
        ids = np.concatenate(ids)
        decisions += len(ids)
        groups, finished = env.step(ids, np.concatenate(choices))
        check_groups(groups, 16)
        for f in finished:
            assert len(f["scores"]) == f["n_players"] == len(f["model_seats"])
            assert f["turns"] > 0 and f["winners"]
        finished_total += len(finished)
        if not groups:
            break
    assert finished_total > 0 and decisions > 16


def test_vecenv_bad_choice():
    env = pe.VecEnv(2, sizes=[1], threads=2)
    g = env.observe()[0]
    with pytest.raises(IndexError):
        env.step(g["game"][:1], np.array([10**6], dtype=np.int32))


def test_vecenv_throughput(capsys):
    env = pe.VecEnv(512, seed_start=100, model_seats="rotate", threads=24)
    groups = env.observe()
    rng = np.random.default_rng(1)
    t0, n = time.time(), 0
    while time.time() - t0 < 5:
        ids = np.concatenate([g["game"] for g in groups])
        ch = np.concatenate([(rng.random(len(g["game"])) * np.diff(g["act_off"])).astype(np.int32) for g in groups])
        groups, _ = env.step(ids, ch)
        n += len(ids)
    with capsys.disabled():
        print(f"\nVecEnv(512 games, random model seat): {n / (time.time() - t0):.0f} decisions/s")
