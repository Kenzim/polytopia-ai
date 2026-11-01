"""The Rust rollout collector (docs/collector.md) records the same rollout as the Python one.

Everything runs on the real engine module with the position-deterministic stand-in policy (no network): one model arm in
lockstep gives identical stores row for row; several retail arms give identical games."""

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from polyai.rl import Collector, Pool, RLConfig, compute_gae, ppo_update, train
from polyai.rl_pool import EntryInfo, NetPolicy, plan_quotas
from polyai.standin import StandInPolicy
from test_rl import CPU, init_run, read_metrics, small_cfg, tiny_model  # noqa: F401

HAVE_ENGINE = importlib.util.find_spec("polyai_engine") is not None
pytestmark = pytest.mark.skipif(not HAVE_ENGINE, reason="polyai_engine is not built")

ONLY_BC = "bc=1,retail:easy=0,retail:normal=0,retail:hard=0,retail:crazy=0"
ONLY_RETAIL = "bc=0,retail:easy=0,retail:normal=1,retail:hard=1,retail:crazy=0"


def collector_classes():
    from polyai.rl_collect import RustCollector

    return Collector, RustCollector


def make(cls, tmp_path=None, *, workers=1, snapshots=0, seed=0, log=None, **kw):
    import polyai_engine as pe

    base = dict(mode="might", games=8, rollout_steps=400, sizes="tiny", threads=3, torch_threads=1, max_lead=0,
                pool_weights=ONLY_BC)
    base.update(kw)
    cfg = RLConfig(**base)
    pool = Pool(cfg, StandInPolicy("bc", salt=2), EntryInfo("bc", "bc"))
    for k in range(snapshots):
        pool.add_snapshot(f"self:u{k:05d}", StandInPolicy(f"s{k}", salt=3 + k), EntryInfo(f"self:u{k:05d}", f"s{k}"))
    ex = ThreadPoolExecutor(max_workers=workers)
    col = cls(cfg, pe, StandInPolicy("learner", salt=1), pool, np.random.default_rng(seed), ex, log)
    col.set_quotas(plan_quotas(pool.weights(), cfg.games, col.seat_p, col.shards))
    return col, ex


def play(cls, rollouts=1, **kw):
    col, ex = make(cls, **kw)
    out = []
    try:
        for u in range(rollouts):
            col.update = u
            col.collect(col.cfg.rollout_steps)
            out.append(({hw: st.finalize() for hw, st in col.stores.items()}, list(col.results), col.n_learner, col.n_opp))
            col.set_quotas(plan_quotas(col.pool.weights(), col.cfg.games, col.seat_p, col.shards))
    finally:
        ex.shutdown()
    return out, col


def assert_same_stores(a: dict, b: dict):
    assert list(a) == list(b)
    for hw in a:
        assert sorted(a[hw]) == sorted(b[hw])
        for k in a[hw]:
            x, y = np.asarray(a[hw][k]), np.asarray(b[hw][k])
            assert x.dtype == y.dtype and x.shape == y.shape, (hw, k, x.dtype, y.dtype, x.shape, y.shape)
            if k in ("adv", "ret"):
                np.testing.assert_allclose(x, y, rtol=0, atol=1e-6, err_msg=f"{hw} {k}")
            else:
                np.testing.assert_array_equal(x, y, err_msg=f"{hw} {k}")


def strip(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k != "weight"}  # the weight depends on the order games finish in


def test_gae_matches_python():
    import polyai_engine as pe

    rng = np.random.default_rng(3)
    for terminal in (True, False):
        for n in (1, 2, 5, 40):
            v = rng.random(n)
            a, r = compute_gae(v, 1.0, terminal, 1.0, 0.95)
            b, s = pe.compute_gae_rust(list(v), 1.0, terminal, 1.0, 0.95)
            np.testing.assert_allclose(a, b, atol=1e-6)
            np.testing.assert_allclose(r, s, atol=1e-6)
            assert len(a) == len(b)


@pytest.mark.parametrize("snapshots,rollouts", [(0, 1), (2, 2)])
def test_one_model_arm_in_lockstep_gives_identical_rollouts(snapshots, rollouts):
    py_cls, rust_cls = collector_classes()
    kw = dict(rollout_steps=1200, games=6, snapshots=snapshots, rollouts=rollouts, turn_limit=8)
    (py, _), (rs, _) = play(py_cls, **kw), play(rust_cls, **kw)
    for (s1, r1, nl1, no1), (s2, r2, nl2, no2) in zip(py, rs):
        assert (nl1, no1) == (nl2, no2)
        assert_same_stores(s1, s2)
        assert [strip(r) for r in r1] == [strip(r) for r in r2]
        assert len(r1) > 0 and sum(len(s["train"]) for s in s1.values()) > 0


def test_features_and_truth_rows_match():
    py_cls, rust_cls = collector_classes()
    kw = dict(rollout_steps=300, games=4, features=True, belief_weight=0.1, sizes="tiny,small")
    (py, _), (rs, _) = play(py_cls, **kw), play(rust_cls, **kw)
    assert "f_tiles" in next(iter(py[0][0].values())) and "truth" in next(iter(py[0][0].values()))
    assert_same_stores(py[0][0], rs[0][0])


def test_drop_capped_leaves_capped_games_untrained():
    py_cls, rust_cls = collector_classes()
    kw = dict(rollout_steps=1500, games=6, turn_limit=4, drop_capped=True)
    (py, _), (rs, _) = play(py_cls, **kw), play(rust_cls, **kw)
    assert any(r["capped"] for r in py[0][1])
    assert_same_stores(py[0][0], rs[0][0])
    assert [strip(r) for r in py[0][1]] == [strip(r) for r in rs[0][1]]


@pytest.mark.parametrize("pad", [True, False])
def test_mixed_sizes_padded_and_not(pad):
    py_cls, rust_cls = collector_classes()
    kw = dict(rollout_steps=400, games=6, sizes="tiny,small", pad_sizes=pad)
    (py, _), (rs, _) = play(py_cls, **kw), play(rust_cls, **kw)
    assert_same_stores(py[0][0], rs[0][0])


def test_retail_arms_play_the_same_games():
    py_cls, rust_cls = collector_classes()
    kw = dict(rollout_steps=800, games=16, workers=2, pool_weights=ONLY_RETAIL, max_lead=2, turn_limit=12)
    (py, _), (rs, _) = play(py_cls, **kw), play(rust_cls, **kw)
    a = {(r["seed"], r["learner_seat"], r["opponent"]): strip(r) for r in py[0][1]}
    b = {(r["seed"], r["learner_seat"], r["opponent"]): strip(r) for r in rs[0][1]}
    common = set(a) & set(b)
    assert len(common) >= 4
    assert all(a[k] == b[k] for k in common)


def test_roles_in_use_games_log_and_quota_changes(tmp_path):
    _, rust_cls = collector_classes()
    log = (tmp_path / "games.jsonl").open("w", buffering=1)
    col, ex = make(rust_cls, snapshots=1, games=6, rollout_steps=800, log=log, turn_limit=6,
                   pool_weights="bc=1,retail:normal=1")
    try:
        assert sum(a.n_active for a in col.arms.values()) == 6
        col.collect(800)
        assert col.n_learner >= 800 and col.n_opp > 0 and col.t_step >= 0 and col.t_infer > 0
        assert col.in_use() <= {"bc", "self:u00000"}
        assert len(col.results) > 0
        lines = [json.loads(x) for x in (tmp_path / "games.jsonl").read_text().splitlines()]
        assert lines == col.results
        # fewer games in the next rollout: finished slots are retired, not replayed
        col.set_quotas(plan_quotas(col.pool.weights(), 3, col.seat_p, col.shards))
        col.collect(600)
        assert sum(a.n_active for a in col.arms.values()) == 3
    finally:
        ex.shutdown()


def test_ppo_update_on_the_rust_stores():
    import torch

    from polyai.model import build_model

    _, rust_cls = collector_classes()
    out, col = play(rust_cls, rollout_steps=600, games=6)
    stores = col.stores
    d = next(iter(stores.values())).finalize()
    assert d["tiles"].shape[0] == next(iter(stores.values())).n
    cfg = RLConfig(mode="might", minibatch=64, bf16=False, device="cpu", critic_warmup=0, epochs=1)
    model, ref = build_model("tiny"), build_model("tiny")
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    res = ppo_update(model, ref, opt, stores, cfg, 0, torch.device("cpu"), np.random.default_rng(0))
    assert not res.get("skipped") and np.isfinite(res["policy_loss"])


def test_network_policies_with_seeded_samplers_give_the_same_rollout(tmp_path, init_run):
    """Batches are built in the Python collector's row order, so a seeded network samples the same moves from either."""
    import polyai_engine as pe

    py_cls, rust_cls = collector_classes()
    rolls = []
    for cls in (py_cls, rust_cls):
        cfg = small_cfg(tmp_path, init_run, games=6, rollout_steps=150, sizes="tiny,small", max_lead=0, threads=2,
                        pool_weights=ONLY_BC, turn_limit=10)
        pool = Pool(cfg, NetPolicy(tiny_model(3), "ref@7", CPU, seed=1), EntryInfo("bc", "ref@7"))
        ex = ThreadPoolExecutor(max_workers=1)
        col = cls(cfg, pe, NetPolicy(tiny_model(), "t@0", CPU, seed=5), pool, np.random.default_rng(0), ex)
        try:
            col.set_quotas(plan_quotas(pool.weights(), cfg.games, col.seat_p, col.shards))
            col.collect(cfg.rollout_steps)
        finally:
            ex.shutdown()
        rolls.append(({hw: st.finalize() for hw, st in col.stores.items()}, [strip(r) for r in col.results]))
    assert_same_stores(rolls[0][0], rolls[1][0])
    assert rolls[0][1] == rolls[1][1]


@pytest.mark.parametrize("overlap", [False, True])
def test_rust_collector_run_writes_metrics_and_resumes(tmp_path, init_run, overlap):
    cfg = small_cfg(tmp_path, init_run, collector="rust", overlap=overlap, updates=3, lr=1e-3)
    assert train(cfg)["updates"] == 3
    run = tmp_path / "t"
    recs = [r for r in read_metrics(run / "metrics.jsonl") if r["kind"] == "update"]
    assert [r["update"] for r in recs] == [1, 2, 3]
    for r in recs:
        assert r["learner_decisions"] >= cfg.rollout_steps and r["minibatches"] > 0 and np.isfinite(r["loss"])
    assert (run / "games.jsonl").read_text().strip()
    assert json.loads((run / "config.json").read_text())["train"]["collector"] == "rust"
    cfg2 = small_cfg(tmp_path, init_run, collector="rust", overlap=overlap, updates=4, resume=True)
    assert train(cfg2)["updates"] == 4


def test_unknown_collector_is_refused(tmp_path, init_run):
    with pytest.raises(SystemExit):
        train(small_cfg(tmp_path, init_run, collector="go"))
