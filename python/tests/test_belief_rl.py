"""The belief auxiliary loss inside PPO (docs/belief.md): weight 0 changes nothing, the loss sees hidden tiles only, and
the truth labels never reach the network's inputs."""

import copy
import importlib.util
import inspect
import json

import numpy as np
import pytest
import torch

from polyai import schema as S
from polyai.belief import full_belief_loss, scalar_loss
from polyai.model import BELIEF_PREFIXES, ModelConfig, PolyNet, add_belief, build_model
from polyai.policy import ModelPolicy, collate
from polyai.rl import RLConfig, make_rl_optimizer, parse_args, plan_quotas, ppo_loss, ppo_update, train
from polyai.rl_pool import NetPolicy, merge, take
from polyai.train_bc import load_checkpoint, make_optimizer, save_checkpoint
from test_arena import fake_obs
from test_rl import CPU, init_run, loss_batch, make_collector, read_metrics, small_cfg, tiny_model  # noqa: F401

HAVE_ENGINE = importlib.util.find_spec("polyai_engine") is not None
needs_engine = pytest.mark.skipif(not HAVE_ENGINE, reason="polyai_engine is not built")


def with_truth(obs: dict, seed: int = 0) -> dict:
    """A fake observation plus random truth labels (hidden tiles chosen at random)."""
    rng = np.random.default_rng(seed)
    b, h, w = obs["tiles"].shape[:3]
    mask = (rng.random((b, h, w)) < 0.6).astype(np.uint8)
    truth = np.stack([rng.integers(0, v, (b, h, w)) for v in S.TRUTH_VOCABS], axis=-1).astype(np.int16)
    truth *= mask[..., None].astype(np.int16)
    return {**obs, "truth": truth, "truth_mask": mask, "truth_g": rng.uniform(0, 8, (b, S.TRUTH_G)).astype(np.float32)}


def belief_batch(model, seed=0):
    obs = with_truth(fake_obs(lens=(4, 6, 3, 5), size=11, seed=seed), seed)
    inputs, mb = loss_batch(model, seed=seed)
    mb.update(truth=torch.from_numpy(obs["truth"]), truth_mask=torch.from_numpy(obs["truth_mask"]),
              truth_g=torch.from_numpy(obs["truth_g"]))
    return obs, inputs, mb


# ---- weight 0 is the old behaviour --------------------------------------------------------------------------------


def test_weight_zero_gives_the_same_losses_with_or_without_heads():
    plain = tiny_model()
    headed = add_belief(copy.deepcopy(plain)).eval()
    cfg = RLConfig(device="cpu")
    inputs, mb = loss_batch(plain)
    ref_logits = plain(*inputs).policy_logits.detach() + 0.1
    ref = ppo_loss(plain(*inputs), ref_logits, mb, cfg, kl_coef=0.3)
    _, _, mb_t = belief_batch(plain)  # truth is offered in the batch, but the weight is 0
    out = ppo_loss(headed(*inputs), ref_logits, {**mb, **{k: mb_t[k] for k in ("truth", "truth_mask", "truth_g")}},
                   cfg, kl_coef=0.3)
    assert set(out) == set(ref) and not any(k.startswith("belief") for k in out)
    for k in ref:
        assert float(out[k]) == float(ref[k]), k


@needs_engine
def test_weight_zero_update_is_bit_for_bit_the_old_update(tmp_path, init_run):
    cfg = small_cfg(tmp_path, init_run, games=8, rollout_steps=100, lr=1e-3, critic_warmup=0)
    col, model, ref = make_collector(cfg)
    try:
        col.set_quotas(plan_quotas(col.pool.weights(), cfg.games, col.seat_p, col.shards))
        col.collect(cfg.rollout_steps)
    finally:
        col.executor.shutdown()
    a, b = copy.deepcopy(model), add_belief(copy.deepcopy(model))
    ra = ppo_update(a, ref, make_optimizer(a, cfg), col.stores, cfg, 0, CPU, np.random.default_rng(0))
    rb = ppo_update(b, ref, make_rl_optimizer(b, cfg), col.stores, cfg, 0, CPU, np.random.default_rng(0))
    for k in ("loss", "policy_loss", "value_loss", "entropy", "kl_bc", "approx_kl"):
        assert ra[k] == rb[k], k
    assert ra["grad_norm"] == pytest.approx(rb["grad_norm"], rel=1e-6)  # the sum order of the clip differs by an ulp
    assert not any(k.startswith("belief") for k in rb)
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), n


# ---- the loss ----------------------------------------------------------------------------------------------------


def test_belief_term_is_added_with_its_weight_and_only_hidden_tiles_count():
    model = add_belief(tiny_model(), hidden=8)
    model.eval()
    _, inputs, mb = belief_batch(model)
    out = model(*inputs, belief=True)
    base = ppo_loss(out, None, mb, RLConfig(device="cpu"), kl_coef=0.0)
    cfg = RLConfig(device="cpu", belief_weight=0.5)
    res = ppo_loss(out, None, mb, cfg, kl_coef=0.0)
    assert float(res["loss"]) == pytest.approx(float(base["loss"]) + 0.5 * float(res["belief"]), rel=1e-5)
    for name in S.TRUTH_NAMES + S.TRUTH_G_NAMES:
        assert f"belief_{name}" in res
    # labels on explored tiles are ignored: scrambling them changes nothing
    mask = mb["truth_mask"].bool()
    scrambled = mb["truth"].clone()
    scrambled[~mask] = 5
    res2 = ppo_loss(out, None, {**mb, "truth": scrambled}, cfg, kl_coef=0.0)
    assert float(res2["belief"]) == pytest.approx(float(res["belief"]), rel=1e-6)
    # ... and labels on hidden tiles are not
    changed = mb["truth"].clone()
    terrain = S.TRUTH_NAMES.index("terrain")
    changed[..., terrain][mask] = (changed[..., terrain][mask] + 1) % S.TRUTH_VOCABS[terrain]
    assert float(ppo_loss(out, None, {**mb, "truth": changed}, cfg, kl_coef=0.0)["belief"]) != float(res["belief"])


def test_belief_gradient_reaches_the_trunk_and_the_heads_only():
    model = add_belief(tiny_model(), hidden=8)
    model.eval()
    _, inputs, mb = belief_batch(model)
    cfg = RLConfig(device="cpu", belief_weight=1.0, vf_coef=0.0, ent_coef=0.0)
    res = ppo_loss(model(*inputs, belief=True), None, mb, cfg, kl_coef=0.0, policy_scale=0.0)
    res["loss"].backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert any(n.startswith("blocks.") and g.abs().sum() > 0 for n, g in grads.items())
    for prefix in BELIEF_PREFIXES:
        assert any(n.startswith(prefix) and g.abs().sum() > 0 for n, g in grads.items()), prefix
    assert not any(n.startswith(("policy_mlp", "src_proj", "value_mlp")) and g.abs().sum() > 0
                   for n, g in grads.items())  # the policy and value heads are untouched


def test_scalar_loss_is_a_scaled_mean_squared_error():
    pred = torch.tensor([[0.5, 1.0], [0.0, 0.0]])
    truth_g = torch.tensor([[5.0, 20.0], [0.0, 10.0]])
    total, per = scalar_loss(pred, truth_g)
    assert float(per["opp_cities"]) == pytest.approx(0.0)
    assert float(per["opp_techs"]) == pytest.approx(((1.0 - 2.0) ** 2 + (0.0 - 1.0) ** 2) / 2)
    assert float(total) == pytest.approx(float(per["opp_techs"]))


def test_a_model_without_the_scalar_head_still_trains_on_the_maps():
    model = build_model(ModelConfig(blocks=3, channels=24, policy_hidden=32, value_hidden=32, belief_hidden=8))
    assert model.belief_scalar_head is None
    obs = with_truth(fake_obs(lens=(4, 6, 3, 5), size=11))
    inputs, _ = loss_batch(tiny_model())
    out = model.eval()(*inputs, belief=True)
    assert out.scalars is None
    loss, per = full_belief_loss(out, torch.from_numpy(obs["truth"]), torch.from_numpy(obs["truth_mask"]),
                                 torch.from_numpy(obs["truth_g"]))
    assert set(per) == set(S.TRUTH_NAMES) and torch.isfinite(loss)


# ---- checkpoints and configs ---------------------------------------------------------------------------------------


def test_added_heads_leave_the_policy_and_value_alone_and_configs_stay_old():
    plain = tiny_model()
    headed = add_belief(copy.deepcopy(plain))
    assert "belief_hidden" not in plain.cfg.to_dict() and "belief_scalars" not in plain.cfg.to_dict()
    assert headed.cfg.belief_hidden == 32 and headed.cfg.belief_scalars
    inputs, _ = loss_batch(plain)
    a, b = plain.eval()(*inputs), headed.eval()(*inputs)
    assert torch.equal(a.policy_logits, b.policy_logits) and torch.equal(a.winner_logits, b.winner_logits)
    assert len(b) == 2  # play never gets the belief output unless it asks


def test_checkpoints_with_heads_load_plain_everywhere_and_with_heads_on_request(tmp_path):
    base = tiny_model()
    headed = add_belief(copy.deepcopy(base), hidden=8)
    save_checkpoint(tmp_path / "ckpt_final.pt", headed, 3)
    plain = load_checkpoint(tmp_path / "ckpt_final.pt")
    assert plain.belief_heads is None and plain.belief_scalar_head is None and not plain.cfg.belief_hidden
    full = load_checkpoint(tmp_path / "ckpt_final.pt", belief=True)
    assert full.belief_scalar_head is not None
    obs = fake_obs()
    got = ModelPolicy(tmp_path).choose(obs)  # the loader behind arena, h2h, serve
    want = NetPolicy(base, "ref@0", CPU).choose(obs)
    np.testing.assert_array_equal(got.index, want.index)
    np.testing.assert_allclose(got.logp, want.logp, atol=1e-6)


def test_belief_flags_parse_and_default_to_off():
    cfg = parse_args(["--init", "x"])
    assert cfg.belief_weight == 0.0 and cfg.belief_lr_mult == 1.0
    cfg = parse_args(["--init", "x", "--belief-weight", "0.5", "--belief-hidden", "16", "--belief-lr-mult", "10"])
    assert (cfg.belief_weight, cfg.belief_hidden, cfg.belief_lr_mult) == (0.5, 16, 10.0)


def test_belief_heads_get_their_own_learning_rate():
    model = add_belief(tiny_model(), hidden=8)
    opt = make_rl_optimizer(model, RLConfig(lr=1e-4, belief_lr_mult=10.0, weight_decay=0.01))
    names = {id(p): n for n, p in model.named_parameters()}
    lrs = {names[id(p)]: g["lr"] for g in opt.param_groups for p in g["params"]}
    assert set(lrs) == set(names.values())
    for n, lr in lrs.items():
        assert lr == pytest.approx(1e-3 if n.startswith(BELIEF_PREFIXES) else 1e-4), n


# ---- labels from the engine ---------------------------------------------------------------------------------------


def test_take_and_merge_carry_the_truth_keys():
    a, b = with_truth(fake_obs(lens=(2, 3), seed=1), 1), with_truth(fake_obs(lens=(4,), seed=2), 2)
    m = merge([a, b])
    assert m["truth"].shape[0] == 3 and m["truth_g"].shape == (3, S.TRUTH_G)
    np.testing.assert_array_equal(m["truth"][2], b["truth"][0])
    t = take(a, np.array([1]))
    np.testing.assert_array_equal(t["truth"][0], a["truth"][1])
    np.testing.assert_array_equal(t["truth_g"][0], a["truth_g"][1])


@needs_engine
def test_the_engine_labels_do_not_change_the_observation():
    import polyai_engine as pe

    kw = dict(n_players=[2], mode=pe.MIGHT, threads=1)
    plain, labelled = pe.VecEnv(6, 3, "all", **kw), pe.VecEnv(6, 3, "all", truth=True, **kw)
    for _ in range(3):
        gp, gl = plain.observe(), labelled.observe()
        assert [g["game"].tolist() for g in gp] == [g["game"].tolist() for g in gl]
        for x, y in zip(gp, gl):
            assert "truth" not in x and {"truth", "truth_mask", "truth_g"} <= set(y)
            for k in ("tiles", "globals", "actions", "act_off"):
                np.testing.assert_array_equal(x[k], y[k])
            hidden = y["tiles"][..., S.COLUMN["unexplored"]] == 1
            np.testing.assert_array_equal(y["truth_mask"].astype(bool), hidden)
            assert (y["truth"][~hidden] == 0).all() and y["truth_g"].shape == (len(y["game"]), S.TRUTH_G)
        games = np.concatenate([g["game"] for g in gp]).astype(np.int32)
        picks = np.zeros(len(games), np.int32)
        plain.step(games, picks)
        labelled.step(games, picks)


@needs_engine
def test_truth_never_reaches_the_network_inputs():
    import polyai_engine as pe

    env = pe.VecEnv(6, 5, "all", n_players=[2], mode=pe.MIGHT, threads=1, truth=True)
    [g, *_] = env.observe()
    assert len(collate(g, np.arange(len(g["game"])), CPU)) == 4  # tiles, globals, actions, mask: no truth among them
    assert not [k for k in inspect.signature(PolyNet.forward).parameters if k.startswith("truth")]
    pol = NetPolicy(tiny_model(), "t@0", CPU)
    stripped = {k: v for k, v in g.items() if not k.startswith("truth")}
    scrambled = {**g, "truth": np.zeros_like(g["truth"]) + 3, "truth_g": np.full_like(g["truth_g"], 99.0)}
    base = pol.choose(stripped)
    for other in (g, scrambled):
        got = pol.choose(other)
        np.testing.assert_array_equal(got.index, base.index)
        np.testing.assert_array_equal(got.logp, base.logp)
        np.testing.assert_array_equal(got.value, base.value)


@needs_engine
def test_the_game_wrapper_returns_labels_that_match_the_mask():
    import polyai_engine as pe

    game = pe.Game.from_setup(pe.sample_setup(11, n_players=[2], mode=pe.MIGHT))
    labels, mask, g = game.truth()
    tiles, _ = game.encode()
    assert labels.shape == (*mask.shape, S.TRUTH_K) and g.shape == (S.TRUTH_G,)
    np.testing.assert_array_equal(mask.astype(bool), tiles[..., S.COLUMN["unexplored"]] == 1)
    assert (labels[mask == 0] == 0).all() and mask.any() and labels[mask == 1].any()


@needs_engine
def test_collector_stores_labels_aligned_with_the_observations(tmp_path, init_run):
    cfg = small_cfg(tmp_path, init_run, games=8, rollout_steps=100, belief_weight=0.5)
    col, _, _ = make_collector(cfg)
    try:
        col.set_quotas(plan_quotas(col.pool.weights(), cfg.games, col.seat_p, col.shards))
        col.collect(cfg.rollout_steps)
    finally:
        col.executor.shutdown()
    seen = 0
    for st in col.stores.values():
        if not st.n:
            continue
        d = st.finalize()
        n = d["tiles"].shape[0]
        assert d["truth"].shape == (*d["tiles"].shape[:3], S.TRUTH_K) and d["truth_g"].shape == (n, S.TRUTH_G)
        np.testing.assert_array_equal(d["truth_mask"].astype(bool), d["tiles"][..., S.COLUMN["unexplored"]] == 1)
        seen += n
    assert seen >= cfg.rollout_steps


@needs_engine
def test_ppo_update_with_belief_trains_and_reports_it(tmp_path, init_run):
    cfg = small_cfg(tmp_path, init_run, games=8, rollout_steps=100, lr=1e-3, critic_warmup=0, belief_weight=0.5,
                    belief_hidden=8, belief_lr_mult=5.0)
    col, model, ref = make_collector(cfg)
    try:
        col.set_quotas(plan_quotas(col.pool.weights(), cfg.games, col.seat_p, col.shards))
        col.collect(cfg.rollout_steps)
    finally:
        col.executor.shutdown()
    model = add_belief(model, hidden=8)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    res = ppo_update(model, ref, make_rl_optimizer(model, cfg), col.stores, cfg, 0, CPU, np.random.default_rng(0))
    assert np.isfinite(res["loss"]) and res["belief"] > 0
    for name in S.TRUTH_NAMES + S.TRUTH_G_NAMES:
        assert np.isfinite(res[f"belief_{name}"])
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n.startswith(BELIEF_PREFIXES))


@needs_engine
def test_a_belief_run_writes_checkpoints_that_load_plain_and_resumes(tmp_path, init_run):
    kw = dict(belief_weight=0.5, belief_hidden=8, belief_lr_mult=3.0, overlap=True)
    assert train(small_cfg(tmp_path, init_run, **kw))["updates"] == 2
    run = tmp_path / "t"
    recs = [r for r in read_metrics(run / "metrics.jsonl") if r["kind"] == "update"]
    assert all("belief" in r and r["belief"] > 0 for r in recs)
    assert json.loads((run / "config.json").read_text())["train"]["belief_weight"] == 0.5
    state = torch.load(run / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert state["model_config"]["belief_hidden"] == 8 and state["model_config"]["belief_scalars"]
    assert load_checkpoint(run / "ckpt_final.pt").belief_heads is None
    assert load_checkpoint(run / "pool" / "snap_u00001.pt").belief_heads is None  # opponents are plain networks
    assert train(small_cfg(tmp_path, init_run, updates=3, resume=True, **kw))["updates"] == 3
