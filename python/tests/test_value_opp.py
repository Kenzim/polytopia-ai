"""Value rows from the opponents' side (`--value-opp`, docs/rl.md): in games against our own networks the opponents'
positions are held until the game ends and join the value loss only, with that seat's result as the target."""

import importlib.util
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from polyai.model import add_features
from polyai.rl import Arm, Collector, RLConfig, ValueStore, keep_value_row, parse_args, train, value_logits
from polyai.rl_pool import plan_quotas, take
from test_arena import fake_obs
from test_rl import CPU, init_run, read_metrics, small_cfg, synth_update, tiny_model  # noqa: F401

HAVE_ENGINE = importlib.util.find_spec("polyai_engine") is not None
needs_engine = pytest.mark.skipif(not HAVE_ENGINE, reason="polyai_engine is not built")

POLICY_ONLY = ("policy_mlp.", "src_proj.", "dst_proj.", "kind_embed.", "arg_embed.", "off_embed.")


def opp_rows(n=10, size=11, seed=3) -> dict:
    obs = fake_obs(lens=(3,) * n, size=size, seed=seed)
    ret = np.random.default_rng(seed).choice([0.0, 0.5, 1.0], n).astype(np.float32)
    return {(size, size): {"tiles": obs["tiles"], "globals": obs["globals"], "seat": np.ones(n, np.int8),
                           "turn": np.ones(n, np.int16), "ret": ret}}


def test_value_logits_are_the_winner_logits_of_forward():
    obs = fake_obs(lens=(3, 5, 2), size=11, seed=1)
    from polyai.policy import collate

    tiles, globals_, actions, mask = collate(obs, np.arange(3), CPU)
    model = tiny_model().eval()
    with torch.no_grad():
        torch.testing.assert_close(value_logits(model, tiles, globals_), model(tiles, globals_, actions, mask).winner_logits)
        fm = add_features(tiny_model()).eval()
        for n, p in fm.named_parameters():
            if n.startswith("feat_"):
                torch.nn.init.normal_(p, std=0.3)
        g = torch.Generator().manual_seed(0)
        ft = torch.randint(0, 4, (3, 11, 11, fm.feat_tile_conv.in_channels), generator=g).to(torch.int16)
        fg = torch.rand(3, fm.feat_global_proj.in_features, generator=g)
        fa = torch.zeros(*actions.shape[:2], fm.feat_action_proj.in_features)
        full = fm(tiles, globals_, actions, mask, f_tiles=ft, f_globals=fg, f_actions=fa).winner_logits
        torch.testing.assert_close(value_logits(fm, tiles, globals_, ft, fg), full)
        assert not torch.allclose(value_logits(fm, tiles, globals_), full)  # the features reach the value


def test_keep_value_row_is_a_deterministic_fraction():
    keys = [(seed, seat, k) for seed in (1, 77, 99_999, -5) for seat in (0, 1, 3) for k in range(400)]
    assert all(keep_value_row(*x, 1.0) for x in keys)
    kept = [keep_value_row(*x, 0.25) for x in keys]
    assert kept == [keep_value_row(*x, 0.25) for x in keys]
    assert 0.22 < np.mean(kept) < 0.28
    # a smaller fraction keeps a subset
    assert all(a for a, b in zip(kept, (keep_value_row(*x, 0.1) for x in keys)) if b)


def test_off_leaves_the_update_as_it_was():
    cfg = RLConfig(device="cpu", minibatch=8, bf16=False, lr=1e-3)
    assert cfg.value_opp == 0.0
    m0, r0 = synth_update(cfg)
    m1, r1 = synth_update(cfg, opp=opp_rows())
    for a, b in zip(m0.state_dict().values(), m1.state_dict().values()):
        assert torch.equal(a, b)
    assert r0 == r1 and not any(k.startswith("value_opp") for k in r0)


def test_on_trains_the_value_path_only_and_keeps_the_learner_minibatches():
    from polyai.rl import make_rl_optimizer, ppo_update
    from rl_synth import synthetic_stores

    def run(cfg, opp):
        torch.manual_seed(0)
        model, ref = tiny_model(0), tiny_model(5)
        stores = synthetic_stores(model, groups=6, seed=1)
        rng = np.random.default_rng(0)
        res = ppo_update(model, ref, make_rl_optimizer(model, cfg), stores, cfg, 3, CPU, rng, opp=opp)
        return model, res, rng.random()

    base = dict(device="cpu", minibatch=32, epochs=1, bf16=False, lr=1e-3, grad_clip=1e9, target_kl=0.0)
    rows = opp_rows(n=10)
    m0, r0, d0 = run(RLConfig(**base), None)
    m1, r1, d1 = run(RLConfig(**base, value_opp=2.0), rows)
    assert d0 == d1  # the learner's minibatches drew the same numbers
    for k in ("positions", "positions_kept", "minibatches", "adv_mean", "adv_std", "ret_mean"):
        assert r0[k] == r1[k], k
    assert r1["value_opp_positions"] == 10 and r1["value_opp_trained"] == 10
    assert r1["value_opp_ret_mean"] == pytest.approx(float(rows[(11, 11)]["ret"].mean()))
    assert np.isfinite(r1["value_opp_loss"])
    s0, s1 = m0.state_dict(), m1.state_dict()
    for k in s0:
        if k.startswith(POLICY_ONLY):  # one optimiser step: the value rows give these no gradient
            assert torch.equal(s0[k], s1[k]), k
    assert not torch.equal(s0["value_mlp.2.weight"], s1["value_mlp.2.weight"])
    # a weight of zero with rows present is the update without them
    m2, r2, _ = run(RLConfig(**base, value_opp=0.0), rows)
    assert all(torch.equal(a, b) for a, b in zip(s0.values(), m2.state_dict().values()))
    # a learner that reads the features needs them in the rows
    cfg = RLConfig(**base, value_opp=1.0)
    torch.manual_seed(0)
    fm = add_features(tiny_model(0))
    with pytest.raises(ValueError):
        ppo_update(fm, None, make_rl_optimizer(fm, cfg), synthetic_stores(fm, groups=2, seed=1), cfg, 3, CPU,
                   np.random.default_rng(0), opp=rows)


def test_a_finished_game_gives_each_held_row_its_seats_result():
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        for drop in (False, True):
            col = Collector(RLConfig(value_opp=1.0, value_opp_frac=1.0, drop_capped=drop), None, None, None,
                            np.random.default_rng(0), ex)
            arm = Arm(("model", 0), None, 4)
            obs = fake_obs(lens=(2, 2, 2, 2), size=11)

            def hold(slot):
                for seat in (1, 2, 1):
                    g = {**obs, "game": np.array([slot] * 4, np.int32), "seat": np.full(4, seat, np.int8)}
                    col._hold_opp(arm, take(g, np.array([0])))

            hold(0)
            hold(2)
            assert arm.opp_seen.tolist() == [3, 0, 3, 0] and len(arm.opp_rows[0]) == 3
            col._flush_opp(arm, 0, {"capped": False, "winners": [2]})
            col._flush_opp(arm, 2, {"capped": True, "winners": []})
            assert arm.opp_rows == {} and arm.opp_seen.tolist() == [0, 0, 0, 0]
            d = col.opp_stores[(11, 11)].finalize()
            assert d["seat"].tolist() == [1, 2, 1] + ([] if drop else [1, 2, 1])
            assert d["ret"].tolist() == [0.0, 1.0, 0.0] + ([] if drop else [0.5, 0.5, 0.5])
            assert d["tiles"].shape == (len(d["ret"]), 11, 11, obs["tiles"].shape[3]) and d["tiles"].dtype == np.int16
            assert isinstance(col.opp_stores[(11, 11)], ValueStore)
    finally:
        ex.shutdown()


def test_parse_args_value_opp_flags():
    cfg = parse_args(["--init", "x", "--value-opp", "1.5", "--value-opp-frac", "0.5"])
    assert (cfg.value_opp, cfg.value_opp_frac) == (1.5, 0.5)
    with pytest.raises(SystemExit):
        parse_args(["--init", "x", "--value-opp", "1", "--value-opp-frac", "0"])


# ---------------------------------------------------------------------------------------------------------------
# On the engine: both collectors
# ---------------------------------------------------------------------------------------------------------------


def collect_rollouts(cls, rollouts=2, **kw):
    from test_collector_rust import make

    col, ex = make(cls, **kw)
    out = []
    try:
        for u in range(rollouts):
            col.update = u
            col.collect(col.cfg.rollout_steps)
            out.append(({hw: st.finalize() for hw, st in col.stores.items()}, dict(col.opp_data), col.opp_pending,
                        col.n_opp))
            col.set_quotas(plan_quotas(col.pool.weights(), col.cfg.games, col.seat_p, col.shards))
    finally:
        ex.shutdown()
    return out


@needs_engine
@pytest.mark.parametrize("features", [False, True])
def test_both_collectors_give_the_same_value_rows_and_the_same_rollout(features):
    from test_collector_rust import assert_same_stores, collector_classes

    py_cls, rust_cls = collector_classes()
    kw = dict(rollout_steps=600, games=6, turn_limit=8, sizes="tiny,small", snapshots=1, features=features,
              credit_per_turn=True)
    py = collect_rollouts(py_cls, value_opp=1.0, value_opp_frac=0.5, **kw)
    rs = collect_rollouts(rust_cls, value_opp=1.0, value_opp_frac=0.5, **kw)
    off = collect_rollouts(py_cls, **kw)
    for (s1, o1, p1, n1), (s2, o2, p2, n2), (s0, o0, p0, n0) in zip(py, rs, off):
        assert_same_stores(s1, s2)
        assert_same_stores(s1, s0)  # the learner's rows do not depend on the option
        assert_same_stores(o1, o2)
        assert (p1, n1) == (p2, n2) and n1 == n0 and o0 == {} and p0 == 0
    rows = [d for _, o, _, _ in py for d in o.values()]
    assert rows and sum(len(d["ret"]) for d in rows) > 0
    for d in rows:
        assert set(np.unique(d["ret"]).tolist()) <= {0.0, 0.5, 1.0}
        assert ("f_tiles" in d) == features and "actions" not in d


@needs_engine
@pytest.mark.parametrize("collector", ["python", "rust"])
def test_with_every_row_kept_each_opponent_decision_is_released_or_held(collector):
    from test_collector_rust import collector_classes

    cls = collector_classes()[collector == "rust"]
    out = collect_rollouts(cls, rollouts=3, rollout_steps=500, games=6, turn_limit=8, value_opp=1.0,
                           value_opp_frac=1.0)
    released = sum(len(d["ret"]) for _, o, _, _ in out for d in o.values())
    assert released > 0 and released + out[-1][2] == sum(n for *_, n in out)


@needs_engine
@pytest.mark.parametrize("collector", ["python", "rust"])
def test_a_run_with_value_opp_logs_the_rows(tmp_path, init_run, collector, capsys):
    cfg = small_cfg(tmp_path, init_run, collector=collector, value_opp=1.0, value_opp_frac=0.5, features=True,
                    credit_per_turn=True, updates=3, rollout_steps=300, turn_limit=10, lr=1e-3,
                    pool_weights="bc=1,retail:easy=0,retail:normal=1,retail:hard=0,retail:crazy=0")
    assert train(cfg)["updates"] == 3
    recs = [r for r in read_metrics(tmp_path / "t" / "metrics.jsonl") if r["kind"] == "update"]
    for r in recs:
        assert r["value_opp_positions"] >= 0 and r["value_opp_pending"] >= 0
    assert sum(r["value_opp_positions"] for r in recs) > 0
    assert any(r["value_opp_trained"] > 0 and np.isfinite(r["value_opp_loss"]) for r in recs)
    assert "vopp" in capsys.readouterr().out
