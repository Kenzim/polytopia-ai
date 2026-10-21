"""`polyai.search`: spec parsing, the contract of SearchPolicy, and the arena loop with single games (needs the engine)."""

import numpy as np
import pytest
import torch

from polyai.arena import ModelAgent, Suite, play_pairing
from polyai.model import build_model
from polyai.policy import ModelPolicy, Policy, observe_games
from polyai.search import (
    END_TURN,
    SearchConfig,
    SearchPolicy,
    _terminal_value,
    play_games,
    play_search_pairing,
    search_agent_name,
    split_search_spec,
)
from polyai.train_bc import save_checkpoint

pe = pytest.importorskip("polyai_engine")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    run = tmp_path_factory.mktemp("runs") / "tinyrun"
    run.mkdir()
    torch.manual_seed(0)
    model = build_model("tiny")
    for p in model.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.3)
    save_checkpoint(run / "ckpt_final.pt", model, 77)
    return run / "ckpt_final.pt"


def to_decision(seed: int, seat: int = 0, advance: int = 0):
    """A tiny 2-player game at the first decision of `seat`, or `advance` retail-AI commands later."""
    g = pe.Game.sample(seed, n_players=[2], sizes=[1], presets=[3])
    while g.current_seat() != seat:
        g.play_ai_move()
    for _ in range(advance):
        g.step(g.ai_move())
    return g


def test_spec_parsing():
    assert split_search_spec("/x/run") == ("/x/run", None)
    spec, cfg = split_search_spec("/x/run:search=6")
    assert spec == "/x/run" and cfg == SearchConfig(k=6)
    spec, cfg = split_search_spec("model:/x/run:sample=0.5:search=4:w=2:skip=0.8:budget=50:turns=0:plan=0")
    assert spec == "model:/x/run:sample=0.5"
    assert cfg == SearchConfig(k=4, w=2.0, skip=0.8, budget=50, turns=0, plan=False)
    assert split_search_spec("/x/run:greedy:search=2")[0] == "/x/run:greedy"
    with pytest.raises(ValueError):
        split_search_spec("/x/run:w=2")
    with pytest.raises(ValueError):
        SearchConfig(k=0)
    assert SearchConfig(k=6).tokens() == ["search=6"]
    assert SearchConfig(k=6, w=2, plan=False).tokens() == ["search=6", "w=2", "plan=0"]


def test_agent_name(checkpoint):
    assert search_agent_name(checkpoint, SearchConfig(k=4), None) == "model:tinyrun:search=4"
    assert search_agent_name(checkpoint, SearchConfig(k=4, turns=2), 0.5) == "model:tinyrun:sample=0.5:search=4:turns=2"


def test_needs_games_and_is_a_policy(checkpoint):
    p = SearchPolicy(ModelPolicy(checkpoint), SearchConfig(k=3))
    assert isinstance(p, Policy) and p.policy_id.endswith("@77+search3")
    g = to_decision(1_000_000)
    with pytest.raises(ValueError):
        p.choose(observe_games([g]))
    with pytest.raises(ValueError):
        p.choose(observe_games([g]), temperature=0.0, games=[g])


def test_k1_is_the_plain_policy(checkpoint):
    base = ModelPolicy(checkpoint, seed=3)
    p = SearchPolicy(ModelPolicy(checkpoint, seed=3), SearchConfig(k=1), seed=3)
    games = [to_decision(1_000_000 + i, advance=i) for i in range(4)]
    obs = observe_games(games)
    for temp in (None, 0.7):
        want, got = base.choose(obs, temperature=temp), p.choose(obs, temperature=temp, games=games)
        np.testing.assert_array_equal(got.index, want.index)
        np.testing.assert_allclose(got.logp, want.logp, atol=1e-6)
        np.testing.assert_allclose(got.value, want.value, atol=1e-6)
        assert [[i for i, _ in t] for t in got.top] == [[i for i, _ in t] for t in want.top]
    assert p.searched == 0 and p.rollout_rows == 0


def test_search_returns_legal_actions_and_leaves_the_game_alone(checkpoint):
    p = SearchPolicy(ModelPolicy(checkpoint), SearchConfig(k=4, skip=1.0, turns=0))
    games = [to_decision(1_000_000 + i, advance=i) for i in range(4)]
    before = [(g.legal_actions().copy(), g.encode()[0].copy(), g.encode()[1].copy(), g.turn, g.scores(),
               g.current_seat()) for g in games]
    obs = observe_games(games)
    lens = np.diff(obs["act_off"])
    for temp in (None, 1.0):
        c = p.choose(obs, temperature=temp, games=games)
        assert (c.index >= 0).all() and (c.index < lens).all()
        assert ((c.value > 0) & (c.value < 1)).all() and (c.logp <= 0).all()
        for t, n in zip(c.top, lens):
            assert 1 <= len(t) <= 3 and all(0 <= j < n for j, _ in t)
    assert p.searched > 0 and p.rollout_rows > 0
    for g, (acts, tiles, glob, turn, scores, seat) in zip(games, before):
        np.testing.assert_array_equal(g.legal_actions(), acts)
        np.testing.assert_array_equal(g.encode()[0], tiles)
        np.testing.assert_array_equal(g.encode()[1], glob)
        assert (g.turn, g.scores(), g.current_seat()) == (turn, scores, seat)
        g.step(0)  # and the live game is still playable


def test_value_is_the_movers_value_at_the_end_of_the_chosen_line(checkpoint):
    """Replay the chosen candidate by hand: clone, play it, then the plain greedy policy to its EndTurn; the value
    head's answer there (the mover is us) is what the search reports."""
    base = ModelPolicy(checkpoint)
    p = SearchPolicy(ModelPolicy(checkpoint), SearchConfig(k=3, skip=1.0, turns=0, budget=500))
    for seed, seat in ((1_000_000, 0), (1_000_001, 1)):
        g = to_decision(seed, seat)
        c = p.choose(observe_games([g]), games=[g])
        chosen = int(c.index[0])
        sim = g.clone()
        if sim.legal_actions()[chosen][0] == END_TURN:
            want = float(base.choose(observe_games([g])).value[0])
        else:
            sim.step(chosen)
            while True:
                assert sim.current_seat() == seat and not sim.is_over()
                ch = base.choose(observe_games([sim]))
                if sim.legal_actions()[ch.index[0]][0] == END_TURN:
                    want = float(ch.value[0])
                    break
                sim.step(int(ch.index[0]))
        assert float(c.value[0]) == pytest.approx(want, abs=1e-4)


def test_terminal_values_are_from_the_seats_view():
    class Over:
        def __init__(self, winners, capped=False):
            self._w, self._c = winners, capped

        def winners(self):
            return self._w

        def capped(self):
            return self._c

    assert _terminal_value(Over([0]), 0) == 1.0 and _terminal_value(Over([0]), 1) == 0.0
    assert _terminal_value(Over([], True), 0) == 0.5 and _terminal_value(Over([0], True), 0) == 0.5


def test_plan_is_followed_without_the_network_and_equals_the_greedy_line(checkpoint):
    base = ModelPolicy(checkpoint)
    p = SearchPolicy(ModelPolicy(checkpoint), SearchConfig(k=3, skip=1.0, turns=1))
    g = to_decision(1_000_000, advance=3)
    turn, first = g.turn, True
    while g.current_seat() == 0 and g.turn == turn:
        obs = observe_games([g])
        c = p.choose(obs, games=[g])
        if not first:  # after the one search of the turn, the plan is the greedy line
            assert int(c.index[0]) == int(base.choose(obs).index[0])
        first = False
        g.step(int(c.index[0]))
    assert p.searched == 1 and p.followed > 0
    assert p.root_rows == 1  # one root forward for the search, none for the plan steps
    q = SearchPolicy(ModelPolicy(checkpoint), SearchConfig(k=3, skip=1.0, turns=1, plan=False))
    g = to_decision(1_000_000, advance=3)
    for _ in range(4):
        q.choose(observe_games([g]), games=[g])
        g.step(g.ai_move())
    assert q.followed == 0 and q.root_rows == 4


def test_a_stale_plan_is_dropped(checkpoint):
    p = SearchPolicy(ModelPolicy(checkpoint), SearchConfig(k=3, skip=1.0, turns=1))
    g = to_decision(1_000_000, advance=3)
    c = p.choose(observe_games([g]), games=[g])
    g.step(int(c.index[0]))
    st = p._state[id(g)]
    assert st.steps  # the chosen line is kept
    g.step(len(g.legal_actions()) - 1)  # something else than the plan: the position is not where the plan left it
    p.choose(observe_games([g]), games=[g])
    assert p.followed == 0


def test_arena_loop_with_k1_reproduces_the_vecenv_arena(checkpoint):
    suite = Suite(seeds=3, sizes=[1], presets=[3], turn_limit=12, batch_games=2, threads=4)
    agent = ModelAgent(checkpoint, None, torch.device("cpu"))
    want = {(r["seed"], r["swap"]): r for r in play_pairing(agent, "retail:normal", suite, log=lambda m: None)}
    got, stats = play_search_pairing(checkpoint, SearchConfig(k=1), None, "retail:normal", suite, log=lambda m: None,
                                     workers=1, name="model:tinyrun")
    assert len(got) == 6 and stats["decisions"] > 0 and stats["searched"] == 0
    for r in got:
        w = want[(r["seed"], r["swap"])]
        for key in ("winner", "tribes", "setup", "capped", "seat_difficulties", "agents"):
            assert r[key] == w[key], key
        assert r["turns"] == w["turns"] and r["commands"] == w["commands"]


def test_arena_loop_with_search_and_workers(checkpoint):
    suite = Suite(seeds=2, sizes=[1], presets=[3], turn_limit=8, batch_games=2, threads=4)
    cfg = SearchConfig(k=3, skip=1.0, turns=2, budget=40)
    records, stats = play_search_pairing(checkpoint, cfg, None, "retail:easy", suite, log=lambda m: None, workers=2,
                                         inflight=2)
    assert len(records) == 4 and stats["searched"] > 0
    name = search_agent_name(checkpoint, cfg, None)
    for r in records:
        assert r["agents"][r["swap"]] == name and r["over"] and r["winner"] in (None, 0, 1)
    done = {(r["seed"], r["swap"], tuple(r["agents"])) for r in records}
    again, _ = play_search_pairing(checkpoint, cfg, None, "retail:easy", suite, done=done, log=lambda m: None, workers=1)
    assert again == []


def test_play_games_forgets_finished_games(checkpoint):
    suite = Suite(seeds=1, sizes=[1], presets=[3], turn_limit=6)
    p = SearchPolicy(ModelPolicy(checkpoint), SearchConfig(k=2, skip=1.0))
    fin = play_games(p, [(1_000_000, 0), (1_000_000, 1)], suite, 1)
    assert len(fin) == 2 and not p._state
    assert {f["swap"] for f in fin} == {0, 1}
