"""The bridge to the PolytopAI website, offline: a recorded game is replayed through `SiteArm` by a stub policy that
chooses deterministically, with a client that answers from the recording and refuses any other request. (Skipped
without an engine module that has `Game.from_position`.)"""

import gzip
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from polyai import polytopai_state as T
from polyai import schema as S
from polyai.policy import Choices
from polyai.polytopai import SiteArm, SiteClient, SiteError, SiteGame

FIXTURE = Path(__file__).parent / "fixtures" / "polytopai_game.json.gz"
HAVE_ENGINE = importlib.util.find_spec("polyai_engine") is not None
if HAVE_ENGINE:
    import polyai_engine as pe

    HAVE_ENGINE = hasattr(pe.Game, "from_position")
needs_engine = pytest.mark.skipif(not HAVE_ENGINE, reason="polyai_engine with Game.from_position is not built")


class HashPolicy:
    """Chooses row `sum(actions) mod n` of every legal list: fixed by the position, and spread over the kinds."""

    policy_id = "hash@0"

    def choose(self, obs: dict, *, temperature: float | None = None) -> Choices:
        off = np.asarray(obs["act_off"])
        acts = np.asarray(obs["actions"]).astype(np.int64)
        index = np.array([int(acts[a:b].sum()) % (b - a) for a, b in zip(off[:-1], off[1:])], np.int32)
        n = len(index)
        return Choices(index, np.full(n, -1.0, np.float32), np.full(n, 0.4, np.float32), [[] for _ in range(n)])


class ReplayClient:
    """Answers from a recorded game, in order; a call that differs from the recorded one is refused."""

    def __init__(self, trace: list[dict]):
        self.trace, self.at = trace, 0
        self.seconds = {}

    def call(self, name: str, body: dict | None = None, *, repeatable: bool = False) -> dict:
        if self.at >= len(self.trace):
            raise SiteError("the recording is over")
        e = self.trace[self.at]
        if e["call"] != name or (name == "move" and e["sent"] != body):
            raise SiteError(f"call {self.at}: recorded {e['call']} {e['sent']}, got {name} {body}")
        self.at += 1
        return e["view"]

    def close(self) -> None:
        pass


@pytest.fixture(scope="module")
def trace() -> list[dict]:
    with gzip.open(FIXTURE, "rt") as f:
        return json.load(f)


def replay(trace: list[dict], **kw) -> tuple[SiteArm, ReplayClient]:
    client = ReplayClient(trace)
    setup = trace[0]["sent"]
    arm = SiteArm(pe, HashPolicy(), 1, sizes=[setup["size"]], maps=[setup["map_type"]], max_games=1,
                  client_factory=lambda k: client, pause_wait=30.0, **kw)
    arm.resume()
    deadline = time.time() + 120
    while arm.alive and time.time() < deadline:
        time.sleep(0.02)
    arm.pause()
    return arm, client


def test_move_keys_join_the_two_legal_lists():
    legal = [
        {"moveType": 1, "src": 5, "target": 6, "label": "Step: 5 -> 6"},
        {"moveType": 5, "target": 9, "label": "Harvest Fruit at 9"},
        {"moveType": 4, "src": 7, "type": 2, "label": "Train Warrior at 7"},
        {"moveType": 7, "type": 4, "label": "Research Hunting"},
        {"moveType": 10, "label": "End Turn"},
        {"moveType": 3, "type": 999, "src": 5, "label": "Something new"},
    ]
    actions = np.array([
        [T.K_MOVE, 0, 5, 6], [T.K_STAY, 0, 5, -1], [T.K_BUILD, 15, 9, -1], [T.K_TRAIN, 2, 7, -1], [T.K_TRAIN, 3, 7, -1],
        [T.K_RESEARCH, 4, -1, -1], [T.K_END_TURN, 0, -1, -1],
    ], np.int16)
    m = T.match_actions(actions, legal)
    assert [x if x in (True, None) else x["label"] for x in m] == [
        "Step: 5 -> 6", True, "Harvest Fruit at 9", "Train Warrior at 7", None, "Research Hunting", "End Turn"]
    assert T.move_key(legal[-1]) is None  # an ability the engine has no counterpart of is never offered to the policy
    # an ability on the tile under a unit names it as `src`; a move without its fields is skipped, not an error
    assert T.move_key({"moveType": 3, "type": 102, "src": 160, "label": "Clear algae at 160"}) == (T.K_CLEAR_TILE_EFFECT, 160)
    assert T.move_key({"moveType": 1, "label": "Step"}) is None


@needs_engine
def test_a_failing_game_is_dropped_and_the_worker_goes_on(trace):
    """Whatever a game raises, it is counted as aborted with the reason, and nothing of it reaches the update."""

    class Broken(ReplayClient):
        def call(self, name, body=None, *, repeatable=False):
            if self.at == 6:
                raise RuntimeError("boom")
            return super().call(name, body, repeatable=repeatable)

    client = Broken(trace)
    arm = SiteArm(pe, HashPolicy(), 1, sizes=[11], max_games=1, client_factory=lambda k: client, pause_wait=30.0)
    arm.resume()
    while not arm.aborted:
        time.sleep(0.01)
    arm.pause()
    data, results = arm.take()
    arm.close()
    assert not results and arm.finished == 0 and "RuntimeError: boom" in arm.errors[0]
    assert sum(len(d["train"]) for d in data) == 0  # its decisions are stored but none of them trains


def test_client_rejects_a_bad_address():
    with pytest.raises(ValueError):
        SiteClient("polytopai.com")


@needs_engine
def test_every_recorded_position_loads_and_holds_the_move_played(trace):
    """The position before each of our recorded moves loads into the engine with us to move, and the engine's legal list
    holds that move."""
    n = 0
    for before, after in zip(trace, trace[1:]):
        if after["call"] != "move":
            continue
        v = before["view"]
        doc, notes = T.position(v["state"], v["legalMoves"])
        assert not notes
        g = pe.Game.from_position(json.dumps(doc))
        assert g.mover_id == v["play"]["human"] and g.turn == v["state"]["settings"]["turn"]
        tiles, glob = g.encode()
        size = v["state"]["settings"]["size"]
        assert tiles.shape == (size, size, S.F_TILE) and glob.shape == (S.G_DIM,)
        keys = {k for a in g.legal_actions() for k in T.action_keys(a)}
        assert T.move_key(after["sent"]) in keys, after["sent"]["label"]
        n += 1
    assert n > 10


@needs_engine
def test_a_position_round_trips_through_the_engine(trace):
    v = trace[0]["view"]
    doc, _ = T.position(v["state"], v["legalMoves"])
    g = pe.Game.from_position(json.dumps(doc))
    again = pe.Game.from_position(g.position())
    assert np.array_equal(g.legal_actions(), again.legal_actions())
    assert np.array_equal(g.encode()[0], again.encode()[0])
    with pytest.raises(ValueError):
        pe.Game.from_position("{}")


@needs_engine
def test_a_stay_needs_no_call_and_ends_the_units_turn(trace):
    """A unit told to stay is done for the turn in the next position, without anything sent to the site."""
    client = ReplayClient(trace)
    game = SiteGame(client, pe, {"tribes": trace[0]["sent"]["tribes"], "map": trace[0]["sent"]["map_type"],
                                 "size": trace[0]["sent"]["size"], "level": 3})
    game.start()
    d = game.decision()
    stays = [i for i, m in enumerate(d.moves) if m is True]
    assert stays, "the first position has a unit that may stay"
    unit = int(d.obs["actions"][stays[0]][2])
    at = client.at
    game.play(d, stays[0])
    assert client.at == at and game.decisions == 1
    after = game.decision().obs["actions"]
    assert not ((after[:, 2] == unit) & np.isin(after[:, 0], [T.K_MOVE, T.K_ATTACK, T.K_STAY])).any()


@needs_engine
def test_a_replayed_game_is_stored_like_a_rollout(trace):
    arm, client = replay(trace, credit_per_turn=True)
    data, results = arm.take()
    arm.close()
    assert not arm.errors and client.at == len(trace)  # every recorded call was made again, in order
    assert len(results) == 1 and arm.finished == 1 and arm.aborted == 0
    r = results[0]
    final = trace[-1]["view"]
    assert r["turns"] == final["state"]["settings"]["turn"] and r["opponent"] == "site:polytopai3"
    assert final["play"]["over"] and not r["capped"]  # the recorded game was played to its end, which we lost
    assert r["outcome"] == float(final["play"]["winner"] == final["play"]["human"]) == 0.0
    (d,) = data
    n = len(d["chosen"])
    assert n == r["decisions"] == arm.decisions > 10
    size = trace[0]["sent"]["size"]
    assert d["tiles"].shape == (n, size, size, S.F_TILE) and d["act_off"][-1] == len(d["actions"])
    lens = np.diff(d["act_off"])
    assert (d["chosen"] >= 0).all() and (d["chosen"] < lens).all()
    # one finished trajectory: every decision trains, towards the outcome
    assert len(d["train"]) == n and (d["traj"] == 0).all() and d["traj_term"].all() and (d["traj_out"] == r["outcome"]).all()
    assert np.allclose(d["ret"], d["adv"] + d["value"]) and (np.diff(d["turn"]) >= 0).all()
    # the moves sent are the recorded ones, so the stays are the decisions that made no call
    moves = sum(e["call"] == "move" for e in trace)
    stays = int((d["actions"][d["act_off"][:-1] + d["chosen"], 0] == T.K_STAY).sum())
    assert n == moves + stays - arm.stats["bridge moves"]


@needs_engine
def test_a_game_at_the_turn_cap_is_a_draw(trace):
    arm, client = replay(trace, turn_cap=4)
    _, results = arm.take()
    arm.close()
    assert client.at < len(trace) and not arm.errors
    assert [(r["capped"], r["outcome"], r["turns"]) for r in results] == [(True, 0.5, 4)]


@needs_engine
def test_a_game_in_flight_is_cut_at_take(trace):
    """`take()` during a game: its decisions so far are a truncated trajectory, whose last one only bootstraps."""
    client = ReplayClient(trace)
    arm = SiteArm(pe, HashPolicy(), 1, sizes=[11], max_games=1, client_factory=lambda k: client, pause_wait=30.0)
    arm.resume()
    while arm.decisions < 5:
        time.sleep(0.005)
    arm.pause()
    data, results = arm.take()
    (d,) = data
    n = len(d["chosen"])
    assert not results and n >= 5 and len(d["train"]) == n - 1 and not d["traj_term"].any()
    arm.resume()  # the game carries on into the next take
    while arm.alive:
        time.sleep(0.02)
    arm.pause()
    data2, results2 = arm.take()
    arm.close()
    assert len(results2) == 1 and sum(len(x["chosen"]) for x in data2) + n == arm.decisions


@needs_engine
def test_the_update_trains_on_site_decisions(trace):
    from polyai.model import build_model
    from polyai.rl import RLConfig, ppo_update

    arm, _ = replay(trace)
    data, _ = arm.take()
    arm.close()
    torch.manual_seed(0)
    model = build_model("tiny")
    cfg = RLConfig(epochs=1, minibatch=16, critic_warmup=0, kl_coef=0.0, kl_coef_end=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    res = ppo_update(model, None, opt, {}, cfg, 0, torch.device("cpu"), np.random.default_rng(0), extra=data)
    assert res["positions"] == sum(len(d["train"]) for d in data) and res["minibatches"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
