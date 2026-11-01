"""`polyai.policy`: the Choices contract of ModelPolicy, and that the arena's ModelAgent plays exactly what it did."""

from pathlib import Path

import numpy as np
import pytest
import torch

from polyai import schema as S
from polyai.arena import ModelAgent
from polyai.model import build_model, eval_pawns
from polyai.policy import Choices, ModelPolicy, Policy, observe_games, resolve_checkpoint
from polyai.train_bc import save_checkpoint
from test_arena import fake_obs

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    run = tmp_path_factory.mktemp("runs") / "tinyrun"
    run.mkdir()
    torch.manual_seed(0)
    model = build_model("tiny")
    for p in model.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.3)
    save_checkpoint(run / "ckpt_final.pt", model, 1234)
    return run / "ckpt_final.pt"


def test_policy_id_and_protocol(checkpoint):
    p = ModelPolicy(checkpoint)
    assert p.policy_id == "tinyrun@1234" and isinstance(p, Policy)
    assert resolve_checkpoint(checkpoint.parent) == checkpoint
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(checkpoint.parent / "nope.pt")


def test_choices_contract(checkpoint):
    obs = fake_obs(lens=(3, 5, 2, 9, 1))
    lens = np.diff(obs["act_off"])
    p = ModelPolicy(checkpoint)
    c = p.choose(obs)
    assert isinstance(c, Choices) and len(c) == 5
    assert c.index.dtype == np.int32 and c.logp.dtype == np.float32 and c.value.dtype == np.float32
    assert (c.index >= 0).all() and (c.index < lens).all()
    assert ((c.value > 0) & (c.value < 1)).all() and (c.logp <= 0).all()
    assert len(c.top) == 5
    for i, t in enumerate(c.top):
        assert len(t) == min(3, lens[i])
        assert all(0 <= j < lens[i] for j, _ in t)
        probs = [q for _, q in t]
        assert probs == sorted(probs, reverse=True) and sum(probs) <= 1 + 1e-5
        assert t[0][0] == c.index[i]  # greedy: the pick is the best move
        assert c.logp[i] == pytest.approx(np.log(t[0][1]), abs=1e-5)
    np.testing.assert_array_equal(c.index, p.choose(obs).index)
    assert eval_pawns(c.value).shape == (5,)


def test_greedy_logp_and_value_match_the_network(checkpoint):
    obs = fake_obs(lens=(4, 7))
    p = ModelPolicy(checkpoint)
    c = p.choose(obs)
    from polyai.policy import collate

    tiles, g, acts, mask = collate(obs, np.arange(2), CPU)
    with torch.no_grad():
        out = p.model(tiles, g, acts, mask)
    lp = torch.log_softmax(out.policy_logits, dim=1)
    np.testing.assert_allclose(c.logp, lp.max(dim=1).values.numpy(), atol=1e-5)
    np.testing.assert_allclose(c.value, torch.softmax(out.winner_logits, -1)[:, 0].numpy(), atol=1e-6)


def test_sampling_reports_the_probability_it_sampled_from(checkpoint):
    obs = fake_obs(lens=(6, 6, 6, 6))
    p = ModelPolicy(checkpoint, seed=3)
    picks = []
    for _ in range(40):
        c = p.choose(obs, temperature=1.0)
        picks.append(c.index)
        for i in range(4):
            top = dict(c.top[i])
            if c.index[i] in top:
                assert c.logp[i] == pytest.approx(np.log(top[c.index[i]]), abs=1e-5)
    assert len(set(map(tuple, picks))) > 1
    hot = p.choose(obs, temperature=5.0)  # a hotter distribution is flatter: the best move gets less
    cold = p.choose(obs, temperature=0.2)
    assert np.mean([t[0][1] for t in hot.top]) < np.mean([t[0][1] for t in cold.top])
    with pytest.raises(ValueError):
        p.choose(obs, temperature=0.0)
    with pytest.raises(ValueError):
        p.choose(obs, temperature=-1.0)


def test_same_seed_same_samples_and_chunking(checkpoint):
    obs = fake_obs(lens=(3, 5, 2, 9, 4, 4))
    a = ModelPolicy(checkpoint, seed=5).choose(obs, temperature=1.0)
    b = ModelPolicy(checkpoint, seed=5).choose(obs, temperature=1.0)
    np.testing.assert_array_equal(a.index, b.index)
    small = ModelPolicy(checkpoint, max_batch=2).choose(obs)
    big = ModelPolicy(checkpoint, max_batch=64).choose(obs)
    np.testing.assert_array_equal(small.index, big.index)
    np.testing.assert_allclose(small.logp, big.logp, atol=1e-5)
    np.testing.assert_allclose(small.value, big.value, atol=1e-5)


def test_one_move_lists_and_empty_lists(checkpoint):
    p = ModelPolicy(checkpoint)
    c = p.choose(fake_obs(lens=(1, 2)))
    assert c.index[0] == 0 and len(c.top[0]) == 1 and c.top[0][0][1] == pytest.approx(1.0)
    bad = fake_obs(lens=(2, 3))
    bad["act_off"] = np.array([0, 2, 2], np.int32)
    with pytest.raises(ValueError):
        p.choose(bad)


def test_agent_is_the_policy_with_the_old_semantics(checkpoint):
    """ModelAgent returns the policy's indices (greedy and sampled, with the same generator stream)."""
    obs = fake_obs(lens=(3, 5, 2, 9))
    agent = ModelAgent(checkpoint, 1.0, CPU, seed=2)
    pol = ModelPolicy(checkpoint, seed=2)
    for _ in range(5):
        np.testing.assert_array_equal(agent.choose(obs), pol.choose(obs, temperature=1.0).index)
    assert agent.step == 1234 and agent.decisions == 20 and agent.infer_seconds > 0


def test_sampling_stream_is_that_of_the_old_arena_agent(checkpoint):
    """The pre-policy ModelAgent drew `multinomial(softmax(logits / T), 1, generator)` per chunk: so does the policy."""
    obs = fake_obs(lens=(3, 5, 2, 9, 4))
    p = ModelPolicy(checkpoint, max_batch=2, seed=9)
    model = p.model
    gen = torch.Generator()
    gen.manual_seed(9)
    from polyai.policy import collate

    for _ in range(3):
        old = np.empty(5, np.int32)
        for s in range(0, 5, 2):
            sel = np.arange(s, min(5, s + 2))
            tiles, g, acts, mask = collate(obs, sel, CPU)
            with torch.inference_mode():
                logits = model(tiles, g, acts, mask).policy_logits.float()
            old[sel] = torch.multinomial(torch.softmax(logits / 0.7, dim=1), 1, generator=gen)[:, 0].numpy()
        np.testing.assert_array_equal(old, p.choose(obs, temperature=0.7).index)


def test_observe_games_matches_the_vecenv_layout(checkpoint):
    pe = pytest.importorskip("polyai_engine")
    games = [pe.Game.sample(s, n_players=[2], sizes=[1]) for s in (1, 2, 3)]
    for g in games:
        for _ in range(6):
            g.play_ai_move()
        while g.current_seat() is None:
            g.play_ai_move()
    obs = observe_games(games)
    n = len(games)
    assert obs["tiles"].shape == (n, 11, 11, S.F_TILE) and obs["globals"].shape == (n, S.G_DIM)
    assert obs["act_off"].shape == (n + 1,) and obs["actions"].shape == (obs["act_off"][-1], 4)
    for i, g in enumerate(games):
        np.testing.assert_array_equal(obs["actions"][obs["act_off"][i]:obs["act_off"][i + 1]], g.legal_actions())
    c = ModelPolicy(checkpoint).choose(obs)
    assert (c.index < np.diff(obs["act_off"])).all()
    with pytest.raises(ValueError):
        observe_games([games[0], pe.Game.sample(1, n_players=[2], sizes=[2])])


def test_tta_views_and_choices(checkpoint):
    """Symmetric views keep the legal-list order; view 0 is the identity; TTA picks legal moves and its choice is the
    same whichever symmetric copy of the position it is given."""
    from polyai.data import transform_actions, transform_tiles
    from polyai.policy import symmetric_views

    obs = fake_obs(lens=(3, 5, 2, 9))
    views = symmetric_views(obs)
    h, w = obs["tiles"].shape[1:3]
    assert len(views) == (8 if h == w else 4)
    np.testing.assert_array_equal(views[0]["tiles"], obs["tiles"])
    np.testing.assert_array_equal(views[0]["actions"], obs["actions"])
    p = ModelPolicy(checkpoint, tta=True)
    c = p.choose(obs)
    lens = np.diff(obs["act_off"])
    assert all(0 <= c.index[i] < lens[i] for i in range(len(lens)))
    assert np.all((c.value >= 0) & (c.value <= 1))
    rot = dict(obs)
    rot["tiles"] = transform_tiles(np.asarray(obs["tiles"]), 2)
    rot["actions"] = transform_actions(np.asarray(obs["actions"]), h, w, 2)
    c2 = p.choose(rot)
    np.testing.assert_array_equal(c.index, c2.index)
    np.testing.assert_allclose(c.value, c2.value, atol=1e-4)
    plain = ModelPolicy(checkpoint).choose(obs)
    single = ModelPolicy(checkpoint, tta=False).choose(obs)
    np.testing.assert_array_equal(plain.index, single.index)
