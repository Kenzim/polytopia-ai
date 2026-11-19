"""The PPO self-play trainer: GAE, the loss, the pool, rollouts on the real engine module and a two-update run
(everything on CPU with a tiny model; the engine tests skip without the module)."""

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

from polyai.model import build_model
from polyai.policy import collate
from polyai.rl import (
    Collector,
    Pool,
    RLConfig,
    WeightEMA,
    anchor_due,
    compute_gae,
    ent_coef_at,
    filter_by_advantage,
    kl_coef_at,
    parse_args,
    ppo_loss,
    ppo_update,
    train,
)
from polyai.rl_pool import (
    EntryInfo,
    NetPolicy,
    OppStats,
    arm_capacity,
    merge,
    merge_padded,
    obs_len,
    pad_bucket,
    parity_adjusted,
    plan_quotas,
    policy_id_of,
    pool_weights,
    retail_player_weights,
    seat_probs,
    take,
)
from polyai.train_bc import save_checkpoint
from rl_synth import synthetic_stores
from test_arena import fake_obs

HAVE_ENGINE = importlib.util.find_spec("polyai_engine") is not None
needs_engine = pytest.mark.skipif(not HAVE_ENGINE, reason="polyai_engine is not built")
CPU = torch.device("cpu")


def tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    m = build_model("tiny")
    for p in m.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.1)
    return m


@pytest.fixture(scope="module")
def init_run(tmp_path_factory) -> Path:
    run = tmp_path_factory.mktemp("rl") / "tinyinit"
    run.mkdir()
    save_checkpoint(run / "ckpt_final.pt", tiny_model(), 7)
    return run


def small_cfg(tmp_path, init_run, **kw) -> RLConfig:
    base = dict(name="t", init=str(init_run), runs_dir=str(tmp_path), updates=2, games=8, rollout_steps=120,
                sizes="tiny", device="cpu", threads=3, torch_threads=1, minibatch=64, critic_warmup=1,
                snapshot_every=1, ckpt_every=1, max_snapshots=2, bf16=False)
    base.update(kw)
    return RLConfig(**base)


# ---------------------------------------------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------------------------------------------


def test_gae_terminal_by_hand():
    v = np.array([0.5, 0.25, 0.75])
    adv, ret = compute_gae(v, final_reward=1.0, terminal=True, gamma=1.0, lam=0.5)
    # deltas: t0 0.25 - 0.5, t1 0.75 - 0.25, t2 1 - 0.75
    d = np.array([-0.25, 0.5, 0.25])
    exp = np.array([d[0] + 0.5 * (d[1] + 0.5 * d[2]), d[1] + 0.5 * d[2], d[2]])
    np.testing.assert_allclose(adv, exp, atol=1e-6)
    np.testing.assert_allclose(ret, exp + v, atol=1e-6)


def test_gae_lambda_one_gives_the_final_reward_as_return():
    v = np.random.default_rng(0).uniform(size=9)
    for outcome in (0.0, 0.5, 1.0):
        _, ret = compute_gae(v, outcome, True, 1.0, 1.0)
        np.testing.assert_allclose(ret, outcome, atol=1e-6)


def test_gae_discounting_and_truncation():
    v = np.array([0.4, 0.6])
    adv, ret = compute_gae(v, 1.0, True, 0.9, 0.0)  # lambda 0: one-step TD
    np.testing.assert_allclose(adv, [0.9 * 0.6 - 0.4, 1.0 - 0.6], atol=1e-6)
    # truncated: the last command is only a bootstrap, so there is one entry fewer and no reward
    adv, ret = compute_gae(np.array([0.4, 0.6, 0.2]), 0.0, False, 1.0, 0.0)
    assert adv.shape == (2,)
    np.testing.assert_allclose(adv, [0.6 - 0.4, 0.2 - 0.6], atol=1e-6)
    assert compute_gae(np.array([0.3]), 0.0, False, 1.0, 0.95)[0].shape == (0,)


def test_gae_per_turn_by_hand():
    v = np.array([0.5, 0.25, 0.75])
    d = np.array([-0.25, 0.5, 0.25])
    # commands 0 and 1 share a turn: no lambda between them
    adv, ret = compute_gae(v, 1.0, True, 1.0, 0.5, turns=np.array([3, 3, 4]))
    exp = np.array([d[0] + (d[1] + 0.5 * d[2]), d[1] + 0.5 * d[2], d[2]])
    np.testing.assert_allclose(adv, exp, atol=1e-6)
    np.testing.assert_allclose(ret, exp + v, atol=1e-6)
    # gamma too applies per turn; two turns between commands count twice. Lambda 0 still sums within a turn.
    adv, _ = compute_gae(np.array([0.4, 0.6, 0.3]), 1.0, True, 0.9, 0.0, turns=np.array([5, 5, 7]))
    d1 = 0.81 * 0.3 - 0.6
    np.testing.assert_allclose(adv, [0.6 - 0.4 + d1, d1, 1.0 - 0.3], atol=1e-6)
    # truncated: same shape rule as per command
    adv, _ = compute_gae(np.array([0.4, 0.6, 0.2]), 0.0, False, 1.0, 0.0, turns=np.array([1, 1, 2]))
    np.testing.assert_allclose(adv, [0.2 - 0.4, 0.2 - 0.6], atol=1e-6)
    # one command per turn is the per-command estimate
    v = np.random.default_rng(1).uniform(size=12)
    for terminal in (True, False):
        a = compute_gae(v, 1.0, terminal, 0.99, 0.95)
        b = compute_gae(v, 1.0, terminal, 0.99, 0.95, turns=np.arange(12))
        np.testing.assert_allclose(a[0], b[0], atol=1e-6)
        np.testing.assert_allclose(a[1], b[1], atol=1e-6)


def test_gae_per_turn_carries_the_result_back_through_a_long_game():
    # 10 turns of 30 commands, a flat value of 0.5 and a win: only the last delta (0.5) is non-zero
    v = np.full(300, 0.5)
    turns = np.repeat(np.arange(1, 11), 30)
    per_cmd, _ = compute_gae(v, 1.0, True, 1.0, 0.95)
    per_turn, _ = compute_gae(v, 1.0, True, 1.0, 0.95, turns=turns)
    assert per_turn[0] == pytest.approx(0.5 * 0.95 ** 9, rel=1e-5)  # nine turn boundaries back to the first command
    assert per_cmd[0] < 1e-6


# ---------------------------------------------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------------------------------------------


def loss_batch(model, bc=None, adv=0.0, seed=0):
    obs = fake_obs(lens=(4, 6, 3, 5), size=11, seed=seed)
    tiles, glob, actions, mask = collate(obs, np.arange(4), CPU)
    model.eval()
    with torch.no_grad():
        out0 = model(tiles, glob, actions, mask)
        lp = torch.log_softmax(out0.policy_logits, 1)
    chosen = torch.tensor([0, 2, 1, 4])
    mb = {"mask": mask, "chosen": chosen, "old_logp": lp.gather(1, chosen[:, None])[:, 0],
          "adv": torch.full((4,), adv), "ret": torch.tensor([1.0, 0.0, 0.5, 1.0])}
    return (tiles, glob, actions, mask), mb


def test_loss_is_sane_at_ratio_one_with_zero_advantage():
    model = tiny_model()
    cfg = RLConfig(ent_coef=0.0, vf_coef=0.0, device="cpu")
    inputs, mb = loss_batch(model)
    out = model(*inputs)
    res = ppo_loss(out, out.policy_logits.detach(), mb, cfg, kl_coef=0.5)
    res["loss"].backward()
    # zero advantage, no entropy or value term, KL to itself: no gradient at all on the policy side
    assert float(res["approx_kl"]) == pytest.approx(0.0, abs=1e-6) and float(res["clip_frac"]) == 0.0
    assert float(res["kl_bc"].detach()) == pytest.approx(0.0, abs=1e-6)
    gmax = max(float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None)
    assert gmax < 1e-5


def test_loss_pushes_the_chosen_action_up_for_positive_advantage():
    model = tiny_model()
    cfg = RLConfig(ent_coef=0.0, vf_coef=0.0, device="cpu")
    inputs, mb = loss_batch(model, adv=1.0)
    out = model(*inputs)
    out.policy_logits.retain_grad()
    res = ppo_loss(out, None, mb, cfg, kl_coef=0.0)
    res["loss"].backward()
    g = out.policy_logits.grad
    for i, c in enumerate(mb["chosen"].tolist()):
        assert g[i, c] < 0  # descent raises the chosen logit
    assert float(res["policy_loss"]) == pytest.approx(-1.0, abs=1e-5)  # ratio 1, advantage 1


def test_loss_clips_and_kl_is_positive_away_from_bc():
    cfg = RLConfig(ent_coef=0.0, vf_coef=0.0, device="cpu", clip=0.2)
    model, bc = tiny_model(0), tiny_model(5)
    inputs, mb = loss_batch(model, adv=1.0)
    out = model(*inputs)
    ref = bc(*inputs).policy_logits.detach()
    res = ppo_loss(out, ref, mb, cfg, kl_coef=1.0)
    assert float(res["kl_bc"]) > 0
    mb2 = dict(mb, old_logp=mb["old_logp"] - 2.0)  # ratio e^2 on a positive advantage: clipped, no gradient
    out2 = model(*inputs)
    res2 = ppo_loss(out2, None, mb2, cfg, kl_coef=0.0)
    assert float(res2["clip_frac"]) == 1.0
    assert float(res2["policy_loss"]) == pytest.approx(-1.2, abs=1e-5)


def test_value_loss_is_cross_entropy_of_the_win_probability():
    model = tiny_model()
    cfg = RLConfig(device="cpu")
    inputs, mb = loss_batch(model)
    out = model(*inputs)
    res = ppo_loss(out, None, mb, cfg, kl_coef=0.0)
    p = torch.softmax(out.winner_logits, 1)[:, 0]
    ret = mb["ret"]
    ref = -(ret * p.log() + (1 - ret) * (1 - p).log()).mean()
    assert float(res["value_loss"]) == pytest.approx(float(ref), rel=1e-4)


def test_kl_coefficient_decays_linearly():
    cfg = RLConfig(kl_coef=0.1, kl_coef_end=0.01, kl_decay_updates=10)
    assert kl_coef_at(0, cfg) == pytest.approx(0.1)
    assert kl_coef_at(5, cfg) == pytest.approx(0.055)
    assert kl_coef_at(50, cfg) == pytest.approx(0.01)


# ---------------------------------------------------------------------------------------------------------------
# Observation helpers, pool and the arm plan
# ---------------------------------------------------------------------------------------------------------------


def test_take_and_merge_roundtrip():
    obs = fake_obs(lens=(3, 5, 2, 4))
    sub = take(obs, np.array([3, 1]))
    assert obs_len(sub) == 2 and sub["act_off"].tolist() == [0, 4, 9]
    np.testing.assert_array_equal(sub["actions"][:4], obs["actions"][10:14])
    np.testing.assert_array_equal(sub["tiles"][1], obs["tiles"][1])
    both = merge([take(obs, np.array([0, 1])), take(obs, np.array([2, 3]))])
    np.testing.assert_array_equal(both["actions"], obs["actions"])
    np.testing.assert_array_equal(both["act_off"], obs["act_off"])


def test_padded_merge_gives_the_same_policy_as_each_size_alone():
    model = tiny_model(3).eval()
    a = fake_obs(lens=(4, 6, 3), size=11, seed=1)
    b = fake_obs(lens=(5, 2), size=14, seed=2)
    both = merge_padded([a, b])
    assert both["tiles"].shape[1:3] == (14, 14) and both["board_mask"][:3, 11:, :].sum() == 0
    assert obs_len(both) == 5
    alone = [NetPolicy(model, "m", CPU).choose(o, temperature=None) for o in (a, b)]
    padded = NetPolicy(model, "m", CPU).choose(both, temperature=None)
    np.testing.assert_array_equal(padded.index, np.concatenate([c.index for c in alone]))
    np.testing.assert_allclose(padded.logp, np.concatenate([c.logp for c in alone]), atol=1e-5)
    np.testing.assert_allclose(padded.value, np.concatenate([c.value for c in alone]), atol=1e-5)
    assert merge_padded([a, a])["tiles"].shape[1:3] == (11, 11) and "board_mask" not in merge_padded([a, a])
    assert pad_bucket((11, 11)) == pad_bucket((16, 16)) != pad_bucket((18, 18)) == pad_bucket((20, 20))
    assert pad_bucket((30, 30)) == 30


def test_parity_adjustment():
    assert parity_adjusted(0.5, 2) == pytest.approx(0.5)
    assert parity_adjusted(0.25, 4) == pytest.approx(0.5)
    assert parity_adjusted(0.0, 3) == 0.0 and parity_adjusted(1.0, 3) == pytest.approx(1.0)


def test_pool_weights_favour_opponents_not_yet_beaten():
    stats = OppStats()
    for _ in range(200):
        stats.record("retail:easy", 2, 1.0)  # always beaten
        stats.record("retail:hard", 2, 0.0)  # never beaten
    assert stats.rate("retail:easy") > 0.9 and stats.rate("retail:hard") < 0.1 and stats.rate("bc") == 0.5
    base = {"retail:easy": 0.2, "retail:hard": 0.2, "bc": 0.2, "self": 0.4}
    w = pool_weights(["retail:easy", "retail:hard", "bc", "self:u1", "self:u2"], base, stats)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["retail:hard"] > w["bc"] > w["retail:easy"]
    assert w["self:u1"] == pytest.approx(w["self:u2"])


def test_length_weights_correct_the_bias_towards_short_games():
    # short games are losses, long ones wins; short ones finish three times as often, so they are three times as
    # many in a window of finished games. Weighting by length recovers the per-game mean of 0.5.
    plain, weighted = OppStats(decay=1.0), OppStats(decay=1.0)
    for _ in range(300):
        for outcome, turns in [(0.0, 20)] * 3 + [(1.0, 60)]:
            plain.record("bc", 2, outcome)
            weighted.record("bc", 2, outcome, turns / 40.0)
    assert plain.raw("bc", 2) == pytest.approx(0.25) and weighted.raw("bc", 2) == pytest.approx(60 / 120)


def test_stats_state_roundtrip():
    s = OppStats()
    s.record("bc", 3, 1.0)
    s.record("bc", 2, 0.5)
    t = OppStats()
    t.load_state_dict(json.loads(json.dumps(s.state_dict())))
    assert t.rate("bc") == pytest.approx(s.rate("bc")) and t.games("bc") == pytest.approx(2.0)


def test_plan_quotas_sum_to_games_and_follow_weights():
    n_players = {2: 55, 3: 25, 4: 20}
    seat_p = seat_probs(n_players)
    assert sum(seat_p) == pytest.approx(1.0) and seat_p[3] < seat_p[2] < seat_p[1] == pytest.approx(seat_p[0])
    assert sum(retail_player_weights(n_players, 3).keys()) == 4  # only 4-player games have a fourth seat
    weights = {"retail:normal": 0.3, "retail:hard": 0.2, "bc": 0.3, "self:u1": 0.2}
    q = plan_quotas(weights, 100, seat_p, 3)
    assert sum(q.values()) == 100
    hard = sum(v for k, v in q.items() if k[0] == "retail" and k[1] == "retail:hard")
    assert hard == 20
    assert abs(sum(v for k, v in q.items() if k[0] == "model") - 50) <= 2  # rounding across arms
    for k, v in q.items():
        assert v <= arm_capacity(k, 100, seat_p, 3)
    # a heavily weighted retail entry is split into arms no bigger than the model shards
    q = plan_quotas({"retail:crazy": 0.9, "bc": 0.1}, 512, seat_p, 16)
    assert sum(q.values()) == 512
    crazy0 = [v for k, v in q.items() if k[:3] == ("retail", "retail:crazy", 0)]
    assert len(crazy0) > 1 and max(crazy0) <= 512 / 16 + 1
    for k, v in q.items():
        assert v <= arm_capacity(k, 512, seat_p, 16)


def test_policy_id():
    assert policy_id_of(Path("/x/runs/bc3_b6c96_v11/ckpt_final.pt"), 4200) == "bc3_b6c96_v11@4200"


# ---------------------------------------------------------------------------------------------------------------
# Rollouts on the real engine
# ---------------------------------------------------------------------------------------------------------------


def make_collector(cfg: RLConfig):
    import polyai_engine as pe

    model = tiny_model()
    ref = tiny_model(3)
    learner = NetPolicy(model, "t@0", CPU)
    ref_pol = NetPolicy(ref, "ref@7", CPU, seed=1)
    pool = Pool(cfg, ref_pol, EntryInfo("bc", "ref@7"))
    snap = NetPolicy(tiny_model(9), "t@0-snap", CPU, seed=2)
    pool.add_snapshot("self:u00001", snap, EntryInfo("self:u00001", "t@1"))
    ex = ThreadPoolExecutor(max_workers=2)
    col = Collector(cfg, pe, learner, pool, np.random.default_rng(0), ex)
    return col, model, ref


@needs_engine
@pytest.mark.parametrize("sizes,max_lead", [("tiny", 0), ("tiny", 4), ("tiny,small", 0), ("tiny,small", 4)])
def test_rollout_shapes_and_logp_match_a_fresh_forward_pass(tmp_path, init_run, sizes, max_lead):
    # with two map sizes the rollout's inference runs on padded batches, the check below on each size alone;
    # max_lead 0 steps the arms in lockstep, 4 lets them run ahead asynchronously
    cfg = small_cfg(tmp_path, init_run, games=10, rollout_steps=150, sizes=sizes, max_lead=max_lead)
    col, model, _ = make_collector(cfg)
    try:
        col.set_quotas(plan_quotas(col.pool.weights(), cfg.games, col.seat_p, col.shards))
        assert sum(a.n_active for a in col.arms.values()) == cfg.games
        col.collect(cfg.rollout_steps)
    finally:
        col.executor.shutdown()
    assert col.n_learner >= cfg.rollout_steps and col.n_opp > 0
    seen = set()
    for hw, st in col.stores.items():
        d = st.finalize()
        n = st.n
        assert d["tiles"].shape[0] == n and d["globals"].shape[0] == n and len(d["act_off"]) == n + 1
        assert d["actions"].shape[0] == d["act_off"][-1]
        lens = np.diff(d["act_off"])
        assert ((d["chosen"] >= 0) & (d["chosen"] < lens)).all()
        assert (d["logp"] <= 0).all() and ((d["value"] > 0) & (d["value"] < 1)).all()
        assert len(d["train"]) > 0 and np.isfinite(d["adv"][d["train"]]).all()
        sel = d["train"][:32]
        tiles, glob, actions, mask = collate(d, sel, CPU)
        with torch.no_grad():
            out = model.eval()(tiles, glob, actions, mask)
        lp = torch.log_softmax(out.policy_logits.float(), 1).gather(1, torch.from_numpy(d["chosen"][sel]).long()[:, None])
        np.testing.assert_allclose(lp[:, 0].numpy(), d["logp"][sel], atol=1e-4)  # PPO ratio is 1 before any update
        seen.add(hw)
    shapes = {k: [g["tiles"].shape[1:3] for g in a.env.observe()] for k, a in col.arms.items()}
    assert len(seen) == len(sizes.split(",")), (seen, shapes)
    for r in col.results:
        assert r["outcome"] in (0.0, 0.5, 1.0) and 2 <= r["n_players"] <= 4
        assert r["opponent_policy_id"]


@needs_engine
def test_small_arms_and_gathered_inference_still_give_the_behaviour_logp(tmp_path, init_run):
    cfg = small_cfg(tmp_path, init_run, games=12, rollout_steps=150, sizes="tiny,small", max_lead=8, arm_games=2,
                    min_infer=6, gather_ms=5.0)
    col, model, _ = make_collector(cfg)
    try:
        assert col.shards == 6  # 12 games / 2 per arm, more than the 2 env threads
        col.set_quotas(plan_quotas(col.pool.weights(), cfg.games, col.seat_p, col.shards))
        assert sum(a.n_active for a in col.arms.values()) == cfg.games
        assert max(a.n_active for a in col.arms.values()) <= 2
        col.collect(cfg.rollout_steps)
    finally:
        col.executor.shutdown()
    assert col.n_learner >= cfg.rollout_steps
    for st in col.stores.values():
        d = st.finalize()
        sel = d["train"][:32]
        tiles, glob, actions, mask = collate(d, sel, CPU)
        with torch.no_grad():
            out = model.eval()(tiles, glob, actions, mask)
        lp = torch.log_softmax(out.policy_logits.float(), 1).gather(1, torch.from_numpy(d["chosen"][sel]).long()[:, None])
        np.testing.assert_allclose(lp[:, 0].numpy(), d["logp"][sel], atol=1e-4)


@needs_engine
def test_ppo_update_runs_and_changes_the_policy(tmp_path, init_run):
    cfg = small_cfg(tmp_path, init_run, games=8, rollout_steps=100, lr=1e-3, critic_warmup=0)
    col, model, ref = make_collector(cfg)
    try:
        col.set_quotas(plan_quotas(col.pool.weights(), cfg.games, col.seat_p, col.shards))
        col.collect(cfg.rollout_steps)
    finally:
        col.executor.shutdown()
    before = [p.detach().clone() for p in model.parameters()]
    from polyai.train_bc import make_optimizer

    extra = [next(st.finalize() for st in col.stores.values() if st.n)]  # the hook for another position source
    n_rollout = sum(len(st.finalize()["train"]) for st in col.stores.values() if st.n)
    res = ppo_update(model, ref, make_optimizer(model, cfg), col.stores, cfg, 0, CPU, np.random.default_rng(0),
                     extra=extra)
    assert res["minibatches"] > 0 and np.isfinite(res["loss"]) and res["kl_bc"] >= 0
    assert res["positions"] == n_rollout + len(extra[0]["train"])
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


@needs_engine
def test_drop_capped_leaves_capped_games_out(tmp_path, init_run):
    cfg = small_cfg(tmp_path, init_run, games=8, rollout_steps=60, turn_limit=2, drop_capped=True)
    col, _, _ = make_collector(cfg)
    try:
        col.set_quotas(plan_quotas(col.pool.weights(), cfg.games, col.seat_p, col.shards))
        col.collect(cfg.rollout_steps)
    finally:
        col.executor.shutdown()
    assert col.results and all(r["capped"] and r["outcome"] == 0.5 for r in col.results)
    trained = sum(len(st.finalize()["train"]) for st in col.stores.values() if st.n)
    # finished (capped) games were dropped; only the truncated tails of unfinished games can be trained on
    assert trained < col.n_learner


# ---------------------------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------------------------


def read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@needs_engine
def test_two_update_run_writes_metrics_and_checkpoints_and_resumes(tmp_path, init_run):
    cfg = small_cfg(tmp_path, init_run)
    summary = train(cfg)
    run = tmp_path / "t"
    assert summary["updates"] == 2
    for name in ("config.json", "metrics.jsonl", "games.jsonl", "ckpt_last.pt", "ckpt_final.pt", "ckpt_u00001.pt"):
        assert (run / name).exists(), name
    assert (run / "pool" / "snap_u00001.pt").exists()
    recs = [r for r in read_metrics(run / "metrics.jsonl") if r["kind"] == "update"]
    assert [r["update"] for r in recs] == [1, 2]
    r = recs[-1]
    for key in ("win_rate", "kl_bc", "entropy", "clip_frac", "value_loss", "decisions_per_sec", "pool_weights",
                "pool_policy_ids", "learner_policy_id"):
        assert key in r, key
    assert r["learner_decisions"] >= cfg.rollout_steps and r["decisions_per_sec"] > 0
    assert json.loads((run / "config.json").read_text())["train"]["name"] == "t"
    state = torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 2 and "optimizer" in state and state["rl"]["snapshots"]

    # resume two more updates: the counters continue and the file is appended to
    cfg2 = small_cfg(tmp_path, init_run, updates=3, resume=True)
    summary2 = train(cfg2)
    assert summary2["updates"] == 3
    recs = [r for r in read_metrics(run / "metrics.jsonl") if r["kind"] == "update"]
    assert [r["update"] for r in recs] == [1, 2, 3]
    assert torch.load(run / "ckpt_final.pt", map_location="cpu", weights_only=False)["step"] == 3


@needs_engine
def test_overlapped_run_trains_on_rollouts_of_the_previous_weights(tmp_path, init_run):
    cfg = small_cfg(tmp_path, init_run, updates=3, overlap=True, lr=1e-3)
    summary = train(cfg)
    run = tmp_path / "t"
    assert summary["updates"] == 3
    recs = [r for r in read_metrics(run / "metrics.jsonl") if r["kind"] == "update"]
    assert [r["update"] for r in recs] == [1, 2, 3]
    # the first rollout is played by the start weights, each later one by the weights before the update alongside it
    assert [r["behaviour_policy_id"] for r in recs] == ["t@0", "t@0", "t@1"]
    for r in recs:
        assert r["learner_decisions"] >= cfg.rollout_steps and r["minibatches"] > 0 and np.isfinite(r["loss"])
    games = [json.loads(line) for line in (run / "games.jsonl").read_text().splitlines()]
    assert {g["update"] for g in games} <= {0, 1, 2}
    assert torch.load(run / "ckpt_final.pt", map_location="cpu", weights_only=False)["step"] == 3

    cfg2 = small_cfg(tmp_path, init_run, updates=4, overlap=True, resume=True)
    assert train(cfg2)["updates"] == 4


def test_resume_without_a_checkpoint_fails(tmp_path, init_run):
    with pytest.raises(SystemExit):
        train(small_cfg(tmp_path, init_run, name="nothing", resume=True))


# ---------------------------------------------------------------------------------------------------------------
# Recipe options: --ema, --ent-schedule, --anchor-every, --adv-filter (defaults leave the update untouched)
# ---------------------------------------------------------------------------------------------------------------


def synth_update(cfg, seed=0, **kw):
    """One ppo_update on synthetic positions with a fixed model, reference and rng; returns (model, metrics)."""
    from polyai.rl import make_rl_optimizer

    torch.manual_seed(0)
    model, ref = tiny_model(0), tiny_model(5)
    stores = synthetic_stores(model, groups=6, seed=1)
    res = ppo_update(model, ref, make_rl_optimizer(model, cfg), stores, cfg, 3, CPU, np.random.default_rng(seed), **kw)
    return model, res


def test_recipe_defaults_keep_the_update_as_it_was():
    d = RLConfig()
    assert (d.ema, d.ent_schedule, d.ent_max, d.anchor_every, d.adv_filter) == (0.0, False, 0.0, 0, 1.0)
    cfg = RLConfig(device="cpu", minibatch=8, bf16=False, lr=1e-3)
    m0, r0 = synth_update(cfg)
    m1, r1 = synth_update(cfg, ema=None, ent_coef=cfg.ent_coef)
    for a, b in zip(m0.state_dict().values(), m1.state_dict().values()):
        assert torch.equal(a, b)
    assert r0 == r1 and r0["positions_kept"] == r0["positions"] == 24 and r0["ent_coef"] == cfg.ent_coef
    assert ent_coef_at(7, cfg, last_entropy=99.0) == cfg.ent_coef  # no schedule: never touched


def test_ema_with_decay_zero_equals_the_live_weights_and_decay_averages():
    cfg = RLConfig(device="cpu", minibatch=8, bf16=False, lr=1e-3)
    torch.manual_seed(0)
    start = tiny_model(0)
    ema0 = WeightEMA(start, 0.0)
    assert all(torch.equal(v, start.state_dict()[k]) for k, v in ema0.state.items())
    model, _ = synth_update(cfg, ema=ema0)
    live = model.state_dict()
    assert set(ema0.state) == set(live)  # parameters and buffers
    for k, v in ema0.state.items():
        assert torch.equal(v, live[k]), k
    assert not torch.equal(live["value_mlp.2.weight"], start.state_dict()["value_mlp.2.weight"])
    # decay 0.5 by hand on a changed parameter, a changed running statistic and an integer buffer
    m = tiny_model(0)
    e = WeightEMA(m, 0.5)
    sd = m.state_dict()
    bufs = [k for k, v in sd.items() if k.endswith("running_mean")]
    ints = [k for k, v in sd.items() if not v.is_floating_point()]
    assert bufs and ints
    before = {k: v.clone() for k, v in sd.items()}
    with torch.no_grad():
        for k in ("value_mlp.2.weight", bufs[0]):
            sd[k].add_(2.0)
        sd[ints[0]].add_(3)
    e.update(m)
    for k in ("value_mlp.2.weight", bufs[0]):
        torch.testing.assert_close(e.state[k], before[k] + 1.0)
    assert torch.equal(e.state[ints[0]], sd[ints[0]])
    # the average does not alias the live tensors
    assert all(e.state[k].data_ptr() != sd[k].data_ptr() for k in sd)


def test_entropy_schedule_decays_linearly_and_drops_to_zero_above_the_threshold():
    cfg = RLConfig(ent_coef=0.01, ent_schedule=True, ent_coef_end=0.0, ent_decay_updates=10)
    assert ent_coef_at(0, cfg) == pytest.approx(0.01)
    assert ent_coef_at(5, cfg) == pytest.approx(0.005)
    assert ent_coef_at(50, cfg) == 0.0
    flat = RLConfig(ent_coef=0.01, ent_schedule=True, ent_max=2.0)  # threshold only: ent_coef_end stays unset
    assert ent_coef_at(100, flat) == pytest.approx(0.01)
    assert ent_coef_at(3, flat, last_entropy=1.9) == pytest.approx(0.01)
    assert ent_coef_at(3, flat, last_entropy=2.1) == 0.0
    assert ent_coef_at(4, flat, last_entropy=1.0) == pytest.approx(0.01)  # back below: the bonus returns
    both = RLConfig(ent_coef=0.01, ent_schedule=True, ent_max=2.0, ent_coef_end=0.0, ent_decay_updates=10)
    assert ent_coef_at(5, both, last_entropy=1.0) == pytest.approx(0.005) and ent_coef_at(5, both, 3.0) == 0.0


def test_entropy_coefficient_reaches_the_loss():
    cfg = RLConfig(device="cpu", minibatch=8, bf16=False)
    _, a = synth_update(cfg, ent_coef=0.0)
    _, b = synth_update(cfg, ent_coef=1.0)
    assert a["ent_coef"] == 0.0 and b["ent_coef"] == 1.0
    assert b["loss"] < a["loss"]  # a large bonus lowers the loss by about its coefficient times the entropy
    inputs, mb = loss_batch(tiny_model())
    out = tiny_model()(*inputs)
    base = ppo_loss(out, None, mb, RLConfig(ent_coef=0.0, vf_coef=0.0), 0.0)
    withent = ppo_loss(out, None, mb, RLConfig(ent_coef=0.0, vf_coef=0.0), 0.0, ent_coef=0.3)
    assert float(base["loss"] - withent["loss"]) == pytest.approx(0.3 * float(base["entropy"]), rel=1e-4)


def test_advantage_filter_keeps_the_largest_magnitudes_across_sizes():
    rng = np.random.default_rng(0)
    data = {}
    for key, n in (("a", 10), ("b", 6)):
        adv = rng.normal(size=n + 3).astype(np.float32)
        data[key] = {"adv_n": adv, "train": np.sort(rng.choice(n + 3, n, replace=False))}
    mags = np.concatenate([np.abs(d["adv_n"][d["train"]]) for d in data.values()])
    top = set(np.sort(mags)[-8:].round(6).tolist())
    kept = filter_by_advantage(data, 0.5)
    got = np.concatenate([np.abs(d["adv_n"][d["train"]]) for d in data.values()])
    assert kept == 8 == len(got) and set(got.round(6).tolist()) == top
    for d in data.values():
        assert (np.diff(d["train"]) > 0).all()  # still sorted indices
    assert filter_by_advantage({"a": {"adv_n": np.arange(5, dtype=np.float32), "train": np.arange(5)}}, 0.01) == 2


def test_advantage_filter_in_the_update():
    cfg = RLConfig(device="cpu", minibatch=8, bf16=False, adv_filter=0.5)
    _, res = synth_update(cfg)
    assert res["positions"] == 24 and res["positions_kept"] == 12
    assert res["minibatches"] == 2 * 2 and res["ret_mean"] == synth_update(RLConfig(device="cpu", minibatch=8, bf16=False))[1]["ret_mean"]
    _, full = synth_update(RLConfig(device="cpu", minibatch=8, bf16=False))
    assert full["minibatches"] == 6 and full["positions_kept"] == 24


def test_anchor_schedule():
    cfg = RLConfig(anchor_every=3)
    assert [n for n in range(1, 10) if anchor_due(n, cfg)] == [3, 6, 9]
    assert not any(anchor_due(n, RLConfig()) for n in range(1, 10))


def test_parse_args_recipe_flags():
    cfg = parse_args(["--init", "x", "--ema", "0.999", "--ent-schedule", "--ent-max", "3.5", "--ent-coef-end", "0",
                      "--ent-decay-updates", "50", "--anchor-every", "20", "--adv-filter", "0.5"])
    assert (cfg.ema, cfg.ent_schedule, cfg.ent_max, cfg.ent_coef_end, cfg.ent_decay_updates, cfg.anchor_every,
            cfg.adv_filter) == (0.999, True, 3.5, 0.0, 50, 20, 0.5)


@needs_engine
def test_anchor_swaps_at_the_right_update_and_ema_checkpoints_load(tmp_path, init_run):
    from polyai.train_bc import load_checkpoint

    cfg = small_cfg(tmp_path, init_run, updates=4, anchor_every=2, ema=0.5, lr=1e-3, critic_warmup=0)
    train(cfg)
    run = tmp_path / "t"
    recs = [r for r in read_metrics(run / "metrics.jsonl") if r["kind"] == "update"]
    # updates 1 and 2 use the BC reference; the copy taken after update 2 is the reference of updates 3 and 4
    assert [r["anchor"] for r in recs] == ["ref", "ref", "self:u00002", "self:u00002"]
    assert recs[2]["kl_bc"] >= 0 and all(np.isfinite(r["loss"]) for r in recs)
    last = torch.load(run / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert last["rl"]["anchor"]["update"] == 4 and "ema" in last  # swapped again after the last update
    final = torch.load(run / "ckpt_final.pt", map_location="cpu", weights_only=False)
    ema_final = torch.load(run / "ckpt_ema_final.pt", map_location="cpu", weights_only=False)
    assert "ema" in final and "ema" not in ema_final
    for k, v in final["ema"].items():
        assert torch.equal(v, ema_final["model"][k])
    assert (run / "ckpt_ema_u00002.pt").exists() and (run / "ckpt_u00002.pt").exists()
    live, avg = load_checkpoint(run / "ckpt_final.pt"), load_checkpoint(run / "ckpt_ema_final.pt")  # plain loaders
    assert live.cfg.to_dict() == avg.cfg.to_dict()
    assert any(not torch.equal(a, b) for a, b in zip(live.parameters(), avg.parameters()))  # decay 0.5 lags the live net
    # resume keeps the anchor and the average
    train(small_cfg(tmp_path, init_run, updates=5, anchor_every=2, ema=0.5, lr=1e-3, critic_warmup=0, resume=True))
    recs = [r for r in read_metrics(run / "metrics.jsonl") if r["kind"] == "update"]
    assert recs[-1]["update"] == 5 and recs[-1]["anchor"] == "self:u00004"


@needs_engine
def test_two_update_smoke_with_all_recipe_options(tmp_path, init_run):
    cfg = small_cfg(tmp_path, init_run, name="all", updates=2, ema=0.9, ent_schedule=True, ent_max=1e-6, ent_coef_end=0.0,
                    ent_decay_updates=4, anchor_every=1, adv_filter=0.5, critic_warmup=0)
    assert train(cfg)["updates"] == 2
    run = tmp_path / "all"
    recs = [r for r in read_metrics(run / "metrics.jsonl") if r["kind"] == "update"]
    assert [r["anchor"] for r in recs] == ["ref", "self:u00001"]
    assert recs[0]["ent_coef"] == pytest.approx(cfg.ent_coef)  # no earlier entropy yet, update 0 of the decay
    assert recs[1]["ent_coef"] == 0.0  # the first update's entropy is above the (tiny) threshold
    for r in recs:
        assert r["positions_kept"] == -(-r["positions"] // 2) and r["minibatches"] > 0 and np.isfinite(r["loss"])
    assert (run / "ckpt_ema_final.pt").exists()


def test_parse_args_defaults_and_flags(tmp_path):
    cfg = parse_args(["--name", "x", "--init", "somewhere", "--updates", "5", "--drop-capped", "--no-bf16"])
    assert cfg.name == "x" and cfg.updates == 5 and cfg.drop_capped and not cfg.bf16
    assert cfg.n_players == "2:55,3:25,4:20" and cfg.gamma == 1.0 and cfg.lam == 0.95
